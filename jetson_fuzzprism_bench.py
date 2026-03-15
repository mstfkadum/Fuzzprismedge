import torch
import torchvision.models as models
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time
import psutil
import os

print("=== INITIALIZING FUZZPRISMEDGE (JETSON NANO CUDA) ===")

# ==========================================
# 1. LOAD MODELS INTO GPU MEMORY
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Target Device: {device}")

print("Loading Tier 2: MobileNet V2 (Lightweight)...")
model_light = models.mobilenet_v2(pretrained=True).to(device)
model_light.eval()

print("Loading Tier 3: ResNet-50 (Heavyweight Proxy for YOLO)...")
model_heavy = models.resnet50(pretrained=True).to(device)
model_heavy.eval()

# Create dummy input tensors (Batch Size 1, 3 Channels, 224x224 and 640x640)
dummy_light = torch.randn(1, 3, 224, 224).to(device)
dummy_heavy = torch.randn(1, 3, 640, 640).to(device)

# --- CUDA WARM-UP ---
print("\n[Warming up CUDA cores (10 iterations)...]")
with torch.no_grad():
    for _ in range(10):
        _ = model_light(dummy_light)
        _ = model_heavy(dummy_heavy)
torch.cuda.synchronize()
print("CUDA Warm-up Complete.")

# ==========================================
# 2. SETUP FUZZY LOGIC CONTROLLER
# ==========================================
battery = ctrl.Antecedent(np.arange(0, 101, 1), 'battery')
motion = ctrl.Antecedent(np.arange(0, 11, 1), 'motion')
trigger = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'trigger')

battery['low'] = fuzz.trimf(battery.universe, [0, 0, 40])
battery['medium'] = fuzz.trimf(battery.universe, [20, 50, 80])
battery['high'] = fuzz.trimf(battery.universe, [60, 100, 100])
motion['weak'] = fuzz.trimf(motion.universe, [0, 0, 4])
motion['moderate'] = fuzz.trimf(motion.universe, [2, 5, 8])
motion['strong'] = fuzz.trimf(motion.universe, [6, 10, 10])

trigger['skip_ai'] = fuzz.trimf(trigger.universe, [0.0, 0.0, 0.4])
trigger['light_ai'] = fuzz.trimf(trigger.universe, [0.2, 0.5, 0.8])
trigger['heavy_ai'] = fuzz.trimf(trigger.universe, [0.6, 1.0, 1.0])

rules = [
    ctrl.Rule(battery['low'] & motion['weak'], trigger['skip_ai']),
    ctrl.Rule(battery['low'] & motion['moderate'], trigger['skip_ai']),
    ctrl.Rule(battery['low'] & motion['strong'], trigger['light_ai']),
    ctrl.Rule(battery['medium'] & motion['weak'], trigger['skip_ai']),
    ctrl.Rule(battery['medium'] & motion['moderate'], trigger['light_ai']),
    ctrl.Rule(battery['medium'] & motion['strong'], trigger['heavy_ai']),
    ctrl.Rule(battery['high'] & motion['weak'], trigger['skip_ai']),
    ctrl.Rule(battery['high'] & motion['moderate'], trigger['heavy_ai']),
    ctrl.Rule(battery['high'] & motion['strong'], trigger['heavy_ai'])
]
trigger_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# ==========================================
# 3. PRECISION BENCHMARK FUNCTION
# ==========================================
def run_precision_benchmark(state_name, bat_val, mot_val, iterations=50):
    print(f"\n[{state_name}] - Bat: {bat_val}%, Mot: {mot_val}/10 (Averaged over {iterations} runs)")
    
    fuzzy_times = []
    ai_times = []
    
    # Reset GPU Memory Tracker
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    for i in range(iterations):
        # 1. Fuzzy Gatekeeper Timing
        t_fuzz_start = time.perf_counter()
        trigger_sim.input['battery'] = bat_val
        trigger_sim.input['motion'] = mot_val
        trigger_sim.compute()
        t_final = trigger_sim.output['trigger']
        t_fuzz_end = time.perf_counter()
        fuzzy_times.append((t_fuzz_end - t_fuzz_start) * 1000)
        
        # 2. Dynamic GPU Routing Timing
        with torch.no_grad():
            if t_final < 0.4:
                # Tier 1: Sleep
                ai_times.append(0.0)
                decision = "TIER 1 (Sleep)"
            elif 0.4 <= t_final < 0.7:
                # Tier 2: MobileNet V2
                torch.cuda.synchronize()
                t_ai_start = time.perf_counter()
                _ = model_light(dummy_light)
                torch.cuda.synchronize()
                t_ai_end = time.perf_counter()
                ai_times.append((t_ai_end - t_ai_start) * 1000)
                decision = "TIER 2 (MobileNet V2)"
            else:
                # Tier 3: Heavyweight (ResNet-50 proxy for YOLO)
                torch.cuda.synchronize()
                t_ai_start = time.perf_counter()
                _ = model_heavy(dummy_heavy)
                torch.cuda.synchronize()
                t_ai_end = time.perf_counter()
                ai_times.append((t_ai_end - t_ai_start) * 1000)
                decision = "TIER 3 (Heavyweight)"

    # Calculate precise averages
    avg_fuzz = sum(fuzzy_times) / iterations
    avg_ai = sum(ai_times) / iterations
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    print(f"  |-- Routing Decision      : {decision}")
    print(f"  |-- Avg Fuzzy Latency     : {avg_fuzz:.2f} ms")
    print(f"  |-- Avg GPU AI Latency    : {avg_ai:.2f} ms")
    print(f"  |-- Total System Latency  : {(avg_fuzz + avg_ai):.2f} ms")
    print(f"  |-- Peak VRAM Fluctuation : {peak_vram:.2f} MB")
    print("-" * 60)

# ==========================================
# 4. EXECUTE TESTS
# ==========================================
print("\n=== EXECUTING 50-ITERATION BENCHMARKS ===")
run_precision_benchmark("EVENT 1: IDLE", bat_val=15, mot_val=3)
run_precision_benchmark("EVENT 2: MODERATE", bat_val=50, mot_val=7)
run_precision_benchmark("EVENT 3: CRITICAL", bat_val=95, mot_val=9)
