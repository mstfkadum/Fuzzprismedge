import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import math

print("=== RUNNING HIGH-PRECISION 24-HOUR BATTERY SIMULATION ===")

# ==========================================
# 1. FUZZPRISMEDGE GATEKEEPER SETUP
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
# 2. JETSON NANO HARDWARE CONSTANTS
# ==========================================
BATTERY_CAPACITY_JOULES = 180000  # Standard 50Wh (10,000mAh at 5V) IoT Battery
IDLE_POWER_W = 2.0                # Jetson Nano Idle Draw
TIER2_POWER_W = 5.0               # Jetson Nano Moderate Draw (MobileNet)
TIER3_POWER_W = 10.0              # Jetson Nano Max Draw (ResNet/YOLO)

# Real Latency Numbers from your Benchmark (converted to seconds per frame)
LATENCY_TIER2_S = 0.040   # ~40 ms
LATENCY_TIER3_S = 0.996   # ~996 ms
FRAMES_PER_EVENT = 30     # Assume each triggered event processes 30 frames to track the object

# ==========================================
# 3. 24-HOUR DIURNAL SIMULATION LOOP
# ==========================================
def run_simulation(mode):
    current_joules = BATTERY_CAPACITY_JOULES
    battery_log = []
    
    for minute in range(1440):
        current_bat_pct = max(0, int((current_joules / BATTERY_CAPACITY_JOULES) * 100))
        battery_log.append(current_bat_pct)
        
        if current_bat_pct == 0:
            continue
            
        # Diurnal Motion: Peaking at midday (lunch rush, high traffic)
        hour = minute / 60
        base_motion = 4 + 5 * math.sin(math.pi * (hour - 6) / 12)
        current_motion = max(0, min(10, int(base_motion + np.random.normal(0, 1.5))))
        
        # Scenario 1: BASELINE (Standard PIR - Always wakes up and runs YOLO)
        if mode == "Baseline":
            if current_motion > 3:  
                active_time = FRAMES_PER_EVENT * LATENCY_TIER3_S
                current_joules -= (TIER3_POWER_W * active_time)
                current_joules -= (IDLE_POWER_W * (60 - active_time))
            else:
                current_joules -= (IDLE_POWER_W * 60)
                
        # Scenario 2: 2-TIER BINARY (Old FuzzEdge - Sleeps or runs YOLO)
        elif mode == "2-Tier":
            trigger_sim.input['battery'] = current_bat_pct
            trigger_sim.input['motion'] = current_motion
            trigger_sim.compute()
            t_final = trigger_sim.output['trigger']
            
            if t_final >= 0.4:
                active_time = FRAMES_PER_EVENT * LATENCY_TIER3_S
                current_joules -= (TIER3_POWER_W * active_time)
                current_joules -= (IDLE_POWER_W * (60 - active_time))
            else:
                current_joules -= (IDLE_POWER_W * 60)
                
        # Scenario 3: 3-TIER FUZZPRISMEDGE (Sleep, MobileNet, or YOLO)
        elif mode == "3-Tier":
            trigger_sim.input['battery'] = current_bat_pct
            trigger_sim.input['motion'] = current_motion
            trigger_sim.compute()
            t_final = trigger_sim.output['trigger']
            
            if t_final < 0.4:
                # Tier 1: Sleep
                current_joules -= (IDLE_POWER_W * 60)
            elif 0.4 <= t_final < 0.7:
                # Tier 2: MobileNet V2
                active_time = FRAMES_PER_EVENT * LATENCY_TIER2_S
                current_joules -= (TIER2_POWER_W * active_time)
                current_joules -= (IDLE_POWER_W * (60 - active_time))
            else:
                # Tier 3: YOLO/ResNet Heavyweight
                active_time = FRAMES_PER_EVENT * LATENCY_TIER3_S
                current_joules -= (TIER3_POWER_W * active_time)
                current_joules -= (IDLE_POWER_W * (60 - active_time))
                
    return battery_log

print("Simulating Baseline (Standard PIR)...")
log_base = run_simulation("Baseline")
print("Simulating 2-Tier Binary Gating...")
log_2tier = run_simulation("2-Tier")
print("Simulating 3-Tier FuzzPrismEdge...")
log_3tier = run_simulation("3-Tier")

# ==========================================
# 4. GENERATE ACADEMIC GRAPH
# ==========================================
plt.figure(figsize=(12, 7))
hours = np.arange(1440) / 60

plt.plot(hours, log_base, label='Baseline: Standard PIR (Always Tier 3)', color='#e74c3c', linewidth=2, linestyle=':')
plt.plot(hours, log_2tier, label='2-Tier Gating: Sleep / Tier 3', color='#f39c12', linewidth=2, linestyle='--')
plt.plot(hours, log_3tier, label='FuzzPrismEdge (3-Tier): Sleep / Tier 2 / Tier 3', color='#2ecc71', linewidth=3)

plt.title('24-Hour Energy Ablation Study: Jetson Nano Battery Discharge', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Time Elapsed (Hours)', fontsize=12, fontweight='bold')
plt.ylabel('Remaining Battery Capacity (%)', fontsize=12, fontweight='bold')
plt.xlim(0, 24)
plt.ylim(0, 105)
plt.xticks(np.arange(0, 25, 2))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='lower left')

# Annotate final percentages
plt.annotate(f'{log_base[-1]}%', xy=(24, log_base[-1]), xytext=(24.2, log_base[-1]), color='#c0392b', fontweight='bold', va='center')
plt.annotate(f'{log_2tier[-1]}%', xy=(24, log_2tier[-1]), xytext=(24.2, log_2tier[-1]), color='#d35400', fontweight='bold', va='center')
plt.annotate(f'{log_3tier[-1]}%', xy=(24, log_3tier[-1]), xytext=(24.2, log_3tier[-1]), color='#27ae60', fontweight='bold', va='center')

plt.tight_layout()
plt.savefig("jetson_battery_ablation.png", dpi=300, bbox_inches='tight')
print("\nGraph saved as 'jetson_battery_ablation.png'!")
print(f"Final Battery - Baseline : {log_base[-1]}%")
print(f"Final Battery - 2-Tier   : {log_2tier[-1]}%")
print(f"Final Battery - 3-Tier   : {log_3tier[-1]}%")
