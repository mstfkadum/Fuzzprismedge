import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import math

print("=== RUNNING RASPBERRY PI 5 BATTERY SIMULATION ===")

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
# 2. RASPBERRY PI 5 HARDWARE CONSTANTS
# ==========================================
BATTERY_CAPACITY_JOULES = 180000  # Standard 50Wh (10,000mAh at 5V) IoT Battery
IDLE_POWER_W = 2.7                # Pi 5 Idle Draw
TIER2_POWER_W = 5.0               # Pi 5 Moderate Draw (MobileNet)
TIER3_POWER_W = 8.0               # Pi 5 Max Draw (100% CPU Utilization)

# Real Latency Numbers from your Pi 5 Benchmark 
LATENCY_TIER2_S = 0.092   # ~92 ms
LATENCY_TIER3_S = 2.419   # ~2419 ms
FRAMES_PER_EVENT = 30     # Assume each triggered event processes 30 frames

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
            
        # Diurnal Motion: Peaking at midday
        hour = minute / 60
        base_motion = 4 + 5 * math.sin(math.pi * (hour - 6) / 12)
        current_motion = max(0, min(10, int(base_motion + np.random.normal(0, 1.5))))
        
        # Scenario 1: BASELINE (Always YOLO)
        if mode == "Baseline":
            if current_motion > 3:  
                active_time = min(60, FRAMES_PER_EVENT * LATENCY_TIER3_S) # Cap at 60s max
                current_joules -= (TIER3_POWER_W * active_time)
                current_joules -= (IDLE_POWER_W * (60 - active_time))
            else:
                current_joules -= (IDLE_POWER_W * 60)
                
        # Scenario 2: 2-TIER BINARY (Sleep or YOLO)
        elif mode == "2-Tier":
            trigger_sim.input['battery'] = current_bat_pct
            trigger_sim.input['motion'] = current_motion
            trigger_sim.compute()
            t_final = trigger_sim.output['trigger']
            
            if t_final >= 0.4:
                active_time = min(60, FRAMES_PER_EVENT * LATENCY_TIER3_S)
                current_joules -= (TIER3_POWER_W * active_time)
                current_joules -= (IDLE_POWER_W * (60 - active_time))
            else:
                current_joules -= (IDLE_POWER_W * 60)
                
        # Scenario 3: 3-TIER FUZZPRISMEDGE
        elif mode == "3-Tier":
            trigger_sim.input['battery'] = current_bat_pct
            trigger_sim.input['motion'] = current_motion
            trigger_sim.compute()
            t_final = trigger_sim.output['trigger']
            
            if t_final < 0.4:
                current_joules -= (IDLE_POWER_W * 60)
            elif 0.4 <= t_final < 0.7:
                active_time = min(60, FRAMES_PER_EVENT * LATENCY_TIER2_S)
                current_joules -= (TIER2_POWER_W * active_time)
                current_joules -= (IDLE_POWER_W * (60 - active_time))
            else:
                active_time = min(60, FRAMES_PER_EVENT * LATENCY_TIER3_S)
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
plt.plot(hours, log_3tier, label='FuzzPrismEdge (3-Tier): Sleep / Tier 2 / Tier 3', color='#8e44ad', linewidth=3)

plt.title('24-Hour Energy Ablation Study: Raspberry Pi 5 Battery Discharge', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Time Elapsed (Hours)', fontsize=12, fontweight='bold')
plt.ylabel('Remaining Battery Capacity (%)', fontsize=12, fontweight='bold')
plt.xlim(0, 24)
plt.ylim(0, 105)
plt.xticks(np.arange(0, 25, 2))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='lower left')

plt.tight_layout()
plt.savefig("pi5_battery_ablation.png", dpi=300, bbox_inches='tight')
print("\nGraph saved as 'pi5_battery_ablation.png'!")

# Find when they hit 0%
def get_death_hour(log):
    try:
        return round(log.index(0) / 60, 1)
    except ValueError:
        return "> 24.0"

print(f"Lifespan - Baseline : {get_death_hour(log_base)} hours")
print(f"Lifespan - 2-Tier   : {get_death_hour(log_2tier)} hours")
print(f"Lifespan - 3-Tier   : {get_death_hour(log_3tier)} hours")
