# Constant-Bias RL Policy — Investigation Summary

## The Problem

After training PPO residual RL to compensate for Stribeck friction, the agent consistently learns a **constant voltage bias** instead of state-dependent stiction compensation. The result is worse than friction LQR alone.

## Experiments (Pre-Fix: Missing Back-EMF)

### Attempt 1: Initial reward tuning (smoothness penalty)
- **Config**: `control_weight=0.05`, `smoothness_weight=0.02`, `residual_scale=4.0`, `wheel_velocity_weight=0.001`
- **Result**: Constant -4V bias, wheel at -10 rad/s, RMS 1.12° vs 1.11° friction LQR
- **Diagnosis**: Smoothness penalty killed chatter but rewarded constant actions. Wheel spinning was nearly free (0.001×100 = 0.1/step).

### Attempt 2: 5D observation + wheel penalty increase
- **Changes**: Added `prev_action` to observation (5D), `wheel_velocity_weight` 0.001→0.01, `ent_coef` 0.01→0.02
- **Result**: Still constant bias (~-3V), wheel at -10 rad/s, RMS 1.15° vs 1.05° friction LQR (-9.3%)
- **Diagnosis**: Cost of spinning (0.01×100=1.0) exactly offset by upright_bonus (1.0). Still a viable strategy.

### Attempt 3: Stronger wheel penalty + reduced authority
- **Changes**: `wheel_velocity_weight` 0.01→0.05, `residual_scale` 4.0→2.0
- **Result**: Wheel now drifts to +12-14 rad/s, RL outputs constant ~-0.8V, RMS 1.37° vs 1.06° friction LQR (-29.3%)
- **Diagnosis**: **Wheel drift happens even in friction LQR alone** (both reach +10-12 rad/s). The penalty hurts the hybrid more because RL adds extra constant torque.

## Root Cause: Missing Back-EMF (Kv=0)

The constant-bias exploit was caused by **missing motor back-EMF** in the simulation.

### The physics gap

The MATLAB model had `Kt=0, Kv=0`, making the motor model unphysical:
- **Without back-EMF**: Motor torque = `Kt * V / Rm` (independent of wheel speed)
- **With back-EMF**: Motor torque = `Kt * (V - Kv*ω) / Rm` (torque decreases with speed)

On real hardware, back-EMF provides **natural velocity-dependent braking**: as the wheel speeds up, back-EMF opposes the driving voltage → less current → less torque → equilibrium. Without it, any net torque accelerates the wheel indefinitely.

### Computing Kv from motor specs

From MATLAB `modelo_pendulo.m`: `w2_noload = 380 RPM`, `i_noload = 0.1 A`, `Rm = 6.667 Ω`

```
ω_noload = 380 × 2π/60 = 39.8 rad/s
Kv = (V - I_noload × Rm) / ω_noload
Kv = (12 - 0.1 × 6.667) / 39.8 ≈ 0.285 V/(rad/s)
```

Back-EMF damping `Kt*Kv/Rm = 0.00744 Nm·s/rad` is **10.6× stronger** than the existing linear damping `b2 = 0.000703 Nm·s/rad`. The simulation was missing the dominant damping mechanism.

### Why this caused constant bias

Without back-EMF, the wheel could accelerate indefinitely with zero cost. The RL discovered that a constant voltage bias keeps the wheel spinning fast → always outside the stiction zone → stiction never blocks the motor → slightly better angle control. The increasing `wheel_velocity_weight` penalties were fighting the symptom, not the cause.

### Why K[1]=0 is correct

`K[1]=0` (zero wheel angle feedback) is standard for reaction wheel systems — wheel angle is irrelevant for balancing. The wheel didn't drift on real hardware because **back-EMF** (not K[1]) is the mechanism that bounds wheel speed.

## Fix: Add Back-EMF to Simulation

### Changes made

1. **`config.py`**: Added `Kv` as computed property from no-load motor specs (`w_noload_rpm=380`, `i_noload=0.1A`)
2. **`reaction_wheel_env.py`**: Changed `self.Kv = 0.0` → `self.Kv = PHYSICAL_PARAMS.Kv` (≈0.285)
3. **Motor torque for stiction**: Updated to `Kt*(12*u - Kv*ω)/Rm` (actual shaft torque)
4. **LQR gains recomputed** via scipy ARE for plant with back-EMF: `K = [-45.0, 0.0, -5.2, -0.62]`
   - Old gains `[-50.0, 0.0, -6.45, -0.34]` were for Kv=0 plant and failed catastrophically with back-EMF
   - K[3] increased from -0.34 to -0.62 (back-EMF already provides most wheel damping)
5. **Reward penalties reduced**: `control_weight` 0.05→0.005, `smoothness_weight` 0.02→0.005, `wheel_velocity_weight` 0.05→0.001
   - With back-EMF bounding wheel speed, aggressive penalties were unnecessary
   - High penalties made any RL action more costly than the angle improvement it provided
6. **Initial conditions focused**: ±0.1 rad (was ±0.3) to train in the stiction-relevant regime

### Updated baselines (with back-EMF)

| Controller | RMS Angle | Wheel Velocity | Notes |
|---|---|---|---|
| No-friction LQR | **0.72°** | ~7 rad/s (bounded) | Target performance |
| Friction LQR (Ts=0.15) | **1.12°** | ~11 rad/s (bounded) | Stiction degradation |

Key improvement: wheel velocity is now **naturally bounded** by back-EMF (was drifting to ±10+ rad/s before).

## Post-Fix Training Results

### v1: Back-EMF + old reward weights
- **Config**: `control_weight=0.05`, `smoothness_weight=0.02`, Kv=0.285, K=[-45,0,-5.2,-0.62]
- **Result**: RMS 1.29° — **worse** than friction LQR (1.12°)
- **Diagnosis**: RL converged to near-zero output (mean |u_RL|=0.42V). The control penalties (0.05/V²) vastly outweigh the tiny per-step angle improvement from overcoming stiction. The RL rationally learns to do nothing. Policy std converged to 0.115 (very narrow, deterministic near-zero policy).

### v2: Back-EMF + reduced penalties + focused ICs
- **Config**: `control_weight=0.005`, `smoothness_weight=0.005`, `wheel_velocity_weight=0.001`, ICs ±0.1 rad, no domain randomization, `ent_coef=0.02`
- **Result**: **RMS 0.92° — 18% improvement over friction LQR**
- **Key metrics**:
  - Mean |u_RL| = 3.54V (actively using authority)
  - Mean signed u_RL = 0.04V (**no constant bias!**)
  - Max wheel velocity = 4.2 rad/s (bounded)
  - Closes 51% of the gap between friction LQR (1.12°) and target (0.72°)
- **Issue**: Policy std was still increasing throughout training (0.6→1.08). The `ent_coef=0.02` is too high relative to the reward signal — the entropy bonus overwhelms the gradient, preventing convergence.

## Next Steps (v3)

1. **Reduce `ent_coef`** from 0.02 → 0.005 to allow policy convergence
2. **Train longer** (2-3M steps) — v2 hadn't converged yet
3. **Re-enable domain randomization** after clean convergence for sim-to-real robustness
4. Gap remaining: 0.92° → 0.72° target (0.20° left to close)
