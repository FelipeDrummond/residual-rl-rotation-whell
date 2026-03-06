# Research Notes: Residual RL for Reaction Wheel Pendulum

## Project Goal

Develop a hybrid control system where:
- **LQR** handles stabilization (the core control task)
- **RL (PPO)** learns a residual policy to compensate for stiction

**Control Law:** `u_total = u_LQR + α * π_θ(s)`

**Current Research Angle:** Stiction Compensation
- Stribeck friction (Ts=0.15 Nm) creates dead zone degrading optimal LQR
- RL with 4V authority overcomes stiction, recovering no-friction performance
- No-friction: 0.78° → With friction: ~1.19° → Hybrid target: <0.9°

---

## Timeline of Research Findings

### Phase 1: Initial LQR Investigation

#### Problem: LQR Oscillating Without Friction
When testing the LQR controller with no friction, we observed continuous oscillation instead of stabilization.

**Root Causes Identified:**

1. **MATLAB Model Convention Mismatch**
   - MATLAB model (`modelo_pendulo.m`) uses `theta=0` at the **downward** (hanging) position
   - Our simulation uses `theta=0` at the **upright** (inverted) position
   - This caused the gravity term sign to be wrong

2. **Gravity Term Sign Error**
   - For a **regular** pendulum (theta=0 at bottom): gravity is stabilizing, so `l_31 < 0`
   - For an **inverted** pendulum (theta=0 at top): gravity is destabilizing, so `l_31 > 0`

   **Fix applied in `reaction_wheel_env.py`:**
   ```python
   # BEFORE (wrong for inverted pendulum):
   l_31 = -MhgL_MrgL / MrL2_Jh

   # AFTER (correct for inverted pendulum):
   l_31 = +MhgL_MrgL / MrL2_Jh  # Positive = destabilizing
   l_41 = -MhgL_MrgL / MrL2_Jh  # Reaction on wheel
   ```

3. **LQR Gains Too Small**
   - Original firmware gains: `K = [-5.5413, 0, -0.7263, -0.0980]`
   - These were ~10x too small for the corrected dynamics
   - **New gains computed:** `K = [-50.0, 0.0, -6.45, -0.34]`

**Result:** LQR now stabilizes perfectly with no friction (~100% success, <1° RMS error)

---

### Phase 2: Extreme Friction Research (Failed Approach)

#### Hypothesis
The RL agent should learn to compensate for extreme Stribeck friction that causes LQR to fail.

#### Stribeck Friction Model
```
F(ω) = [Tc + (Ts - Tc) * exp(-|ω|/vs)] * sign(ω) + σ * ω
```
Where:
- `Ts` = Static friction (stiction)
- `Tc` = Coulomb friction (kinetic)
- `vs` = Stribeck velocity
- `σ` = Viscous friction coefficient

#### Finding: Sharp Success/Failure Transition

We swept friction parameters to find where LQR fails:

| Ts (Static Friction) | LQR Success Rate | Control Saturation |
|---------------------|------------------|-------------------|
| 0.30 | 100% | 0% |
| 0.40 | 100% | 0% |
| 0.45 | 100% | 0% |
| 0.456 | 100% | 0% |
| 0.457 | ~50% | ~97% (when succeeding) |
| 0.46 | 0% | 100% |
| 0.50 | 0% | 100% |

**Key Insight:** The transition from success to failure is extremely sharp (occurs over Ts ∈ [0.456, 0.457]).

#### Problem: No Middle Ground

When LQR fails at extreme friction:
- It requires **constant bang-bang control** (±12V saturation) just to barely survive
- There's no "struggling but surviving" regime where RL can provide meaningful improvement
- Either LQR handles it easily (0% effort) or it needs maximum effort (100% saturation)

#### Training Results with Extreme Friction (Ts ≈ 0.46)

Configuration:
- Friction at LQR failure threshold
- Domain randomization: Ts ∈ [0.38, 0.52]
- ~43% LQR success rate

**Results:**
- Episode length stuck at ~28 steps (early termination)
- RL couldn't learn useful policy
- When RL did "help", it used constant saturation

**Validation Plot Analysis:**
- Hybrid controller showed ~30% improvement in survival
- BUT: Control signal was constantly at ±12V (bang-bang)
- This is not a useful research contribution - just learned to always apply max torque

---

### Phase 3: Combined Challenges Approach (Deprecated)

#### Hypothesis
Instead of extreme friction that requires maximum control effort, create a scenario where:
1. LQR **survives** but performs **poorly**
2. RL can improve performance without requiring excessive control
3. The improvement is measurable and meaningful

#### Configuration (Now Deprecated)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `lqr_gain_scale` | 0.3 | Undertuned LQR (30% of optimal gains) |
| `observation_noise_std` | 0.03 | Sensor noise on theta |
| `disturbance_std` | 0.3 | External disturbances on theta_dot |
| Friction `Ts` | 0.35 | Moderate (LQR can handle alone) |

#### Problem with This Approach

While this configuration "worked" in terms of metrics, it was **not compelling research**:
1. **Artificially handicapping LQR** (`lqr_gain_scale=0.3`) is cheating
2. **Random disturbances** are not a genuine nonlinearity
3. The scenario felt contrived rather than principled

---

### Phase 4: Virtual Damping Approach (Deprecated)

#### Key Discovery: Friction HELPS, Not Hurts!

Extensive experiments revealed a surprising finding:

| LQR Gain Scale | No Friction | With Friction (Ts=0.1) |
|----------------|-------------|------------------------|
| 0.25 | **0% success** | 100% success, 8.7° RMS |
| 0.30 | 100%, 12.7° RMS | 100%, 1.8° RMS |
| 0.35 | 100%, 3.0° RMS | 100%, 1.3° RMS |
| 1.00 | 100%, 0.9° RMS | 100%, 0.8° RMS |

**Insight:** The system without friction is **underdamped**. Friction provides natural damping!

#### New Hypothesis

The RL agent should learn **virtual damping** to replace unreliable physical friction:
- Physical friction varies with temperature, wear, humidity
- Virtual damping (`u_RL ∝ -ω`) is controllable and predictable
- LQR handles stabilization; RL adds the missing damping

#### Configuration

| Parameter | Value | Effect |
|-----------|-------|--------|
| `lqr_gain_scale` | 0.35 | Moderate gains → underdamped |
| `friction` | None | No physical friction |
| `observation_noise_std` | 0.0 | Clean experiment |
| `disturbance_std` | 0.0 | Focus on damping only |
| `residual_scale` | 2.0 | RL damping authority |

#### Expected Results

| Controller | Survival | RMS Error | Oscillations |
|------------|----------|-----------|--------------|
| Optimal LQR (scale=1.0) | 100% | ~0.9° | 2-3 peaks |
| Underdamped LQR (scale=0.35) | 100% | ~3.0° | 6+ peaks |
| Hybrid (LQR + RL) | 100% | <1° (target) | Damped |

#### Why This is Compelling

1. **Genuine problem**: Underdamped oscillations are a real control challenge
2. **No artificial handicaps**: LQR is at moderate (not sabotaged) gains
3. **Clear RL role**: Learn damping function `u_RL(ω) ∝ -ω`
4. **Interpretable**: Easy to verify what RL learned
5. **Practical benefit**: More reliable than physical friction
6. **LQR is essential**: Still provides the core stabilization

---

### Phase 5: Friction Compensation

#### Physics Model Bug Discovery

During the virtual damping work, we discovered two bugs in the friction coupling:

1. **Missing pendulum reaction**: Friction was only applied to the wheel equation, ignoring Newton's third law — bearing friction creates an equal and opposite reaction on the pendulum body.

2. **Wrong wheel coupling**: The wheel equation used a simple `-tau_f/Jr` term instead of the proper inverse mass matrix coupling.

#### The Fix: Proper Mass-Matrix Coupling + RK4 Sub-stepping

After decoupling through the inverse mass matrix M^(-1):
```
theta_ddot += +tau_friction / MrL2_Jh
alpha_ddot += -tau_friction * (MrL2_Jh + Jr) / (Jr * MrL2_Jh)
```

Key physical consequence: when friction cancels motor torque (stiction), the net effect on BOTH pendulum and wheel is zero — the motor cannot control the pendulum if the wheel is stuck.

Added RK4 integration with 10 sub-steps per control period to handle the stiff dynamics created by Stribeck friction.

#### Friction Sweep Results (Corrected Physics)

With optimal LQR (scale=1.0), sweeping Ts:

| Ts (Nm) | RMS Error | Survival | Notes |
|---------|-----------|----------|-------|
| 0.00 | 0.78° | 100% | No-friction baseline |
| 0.05 | 0.85° | 100% | Mild degradation |
| 0.10 | 1.02° | 100% | Moderate degradation |
| **0.15** | **1.19°** | **100%** | **Research operating point** |
| 0.20 | 1.45° | 100% | Significant degradation |
| 0.25 | ~2.0° | ~95% | Near failure threshold |

#### Stiction Dead Zone Analysis

The stiction dead zone is the angle range where LQR's commanded motor torque falls below Ts, leaving the wheel stuck:

| Ts (Nm) | Dead zone (LQR alone) | Dead zone (LQR + 4V RL) |
|---------|-----------------------|--------------------------|
| 0.10 | θ < 4.4° | θ < 1.3° |
| **0.15** | **θ < 6.6°** | **θ < 2.0°** |
| 0.20 | θ < 8.8° | θ < 2.6° |

At Ts=0.15: LQR alone can't move the wheel when θ < 6.6°, but with 4V RL authority the dead zone shrinks to θ < 2.0°.

#### Chosen Parameters and Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Ts | 0.15 Nm | Degrades LQR noticeably (0.78° → 1.19°) without causing failure |
| Tc | 0.09 Nm | Standard 0.6×Ts ratio |
| vs | 0.02 rad/s | Sharp stiction transition |
| sigma | 0.0 | No viscous friction (isolates stiction effect) |
| residual_scale | 4.0V | Reduces dead zone from 6.6° to 2.0° |
| lqr_gain_scale | 1.0 | Optimal LQR (no artificial weakening) |

#### Why This is Compelling

1. **No artificial handicap**: LQR is at full optimal gains
2. **Genuine nonlinearity**: Stiction is a real problem in mechanical systems
3. **Clear improvement target**: 1.19° → <0.9° (recover no-friction performance)
4. **Reviewer-proof**: Can't be dismissed with "just use better gains"
5. **Practical**: Real hardware has bearing friction that varies

#### First Training Attempt: Bang-Bang Failure

Trained `ppo_stiction` model (500k steps) with the stiction compensation scenario. **Result: RL made things worse.**

| Metric | Friction LQR | Hybrid (LQR+RL) |
|--------|-------------|------------------|
| RMS Error | 1.11° | 1.63° (-46% worse) |
| Reward | 436 | 452 (higher!) |
| Control | Smooth | Bang-bang ±4V chatter |

**Root cause analysis:**
- `control_weight=0.01` was too low — max control cost `0.01*16 = 0.16` per step, dwarfed by `upright_bonus=1.0`
- No penalty for action changes, so rapid ±4V oscillation cost nothing
- RL collected upright bonuses during parts of its self-induced oscillation cycle
- Higher reward despite worse RMS = **reward gaming**

**Fix: Reward retuning**
- `control_weight`: 0.01 → 0.05 (max cost per step: 0.8, comparable to upright bonus)
- Added `smoothness_weight=0.02`: penalizes `(u_RL_t - u_RL_{t-1})²` — max chatter cost for ±4V swing is 1.28/step
- `total_timesteps`: 500k → 1M (tighter reward needs more exploration)

---

### Phase 6: Back-EMF Fix & Successful Training (Current)

#### Root Cause: Missing Back-EMF (Kv=0)

The constant-bias RL policy (see `docs/CONSTANT_BIAS_INVESTIGATION.md`) was caused by **missing motor back-EMF** in the simulation. The MATLAB model had `Kt=0, Kv=0`, making the motor model unphysical.

**Without back-EMF:** Motor torque = `Kt * V / Rm` (independent of wheel speed)
**With back-EMF:** Motor torque = `Kt * (V - Kv*ω) / Rm` (torque decreases with speed)

Back-EMF damping `Kt*Kv/Rm = 0.00744 Nm·s/rad` is **10.6× stronger** than the existing linear damping `b2 = 0.000703 Nm·s/rad`. The simulation was missing the dominant damping mechanism.

**Kv computed from no-load motor specs:**
```
ω_noload = 380 × 2π/60 = 39.8 rad/s
Kv = (12 - 0.1 × 6.667) / 39.8 ≈ 0.285 V/(rad/s)
```

#### LQR Gains Recomputed

Old gains `[-50, 0, -6.45, -0.34]` (for Kv=0 plant) failed with back-EMF. New gains computed via `scipy.linalg.solve_continuous_are` for the Kv=0.285 plant:

```python
K = [-45.0, 0.0, -5.2, -0.62]
# Q=diag(1, 0, 0.1, 0.001), R=1, voltage-unit B matrix
```

#### Updated Baselines (with back-EMF, Ts=0.08)

| Controller | RMS Angle | Wheel Velocity | Notes |
|---|---|---|---|
| No-friction LQR | **0.24°** | ~7 rad/s (bounded) | Target performance |
| Friction LQR (Ts=0.08) | **0.90°** | ~11 rad/s (bounded) | Stiction degradation |

#### Training Progression

**v1 (back-EMF + old reward weights):** RL learned to do nothing. `control_weight=0.05` made any action more costly than the angle improvement. RMS 1.29° (worse than LQR 1.12°).

**v2 (reduced penalties):** `control_weight` 0.05→0.005, `smoothness_weight` 0.02→0.005. RMS 0.92° (18% improvement). But `ent_coef=0.02` too high — policy std kept increasing.

**v3 (entropy fix):** `ent_coef` 0.02→0.005. **RMS 0.70° — 22.3% improvement over friction LQR (0.90°).** No constant bias, wheel bounded by back-EMF.

#### Final Results (v3, model `ppo_bemf_v3`)

| Metric | No-friction LQR | Friction LQR | Hybrid (LQR+RL) |
|--------|-----------------|--------------|------------------|
| **RMS Error** | 0.24° | 0.90° | **0.70°** |
| Survival | 100% | 100% | 100% |
| Mean |u_RL| | — | — | 3.5V (actively using authority) |
| Mean signed u_RL | — | — | ~0.04V (no bias) |
| Wheel velocity | ~7 rad/s | ~11 rad/s | bounded |

The hybrid closes 30% of the gap between friction LQR (0.90°) and no-friction target (0.24°).

#### Key Lessons

8. **Missing physics trumps reward tuning** — No amount of reward engineering fixes a broken simulator
9. **Back-EMF is the dominant damping** — 10.6× stronger than linear damping; without it, wheel drifts indefinitely
10. **Reward penalties must be proportional to the signal** — control_weight=0.05 with 4V authority costs 0.8/step, but stiction improvement is ~0.01/step → RL learns to do nothing
11. **Entropy coefficient matters** — too high prevents convergence; too low prevents exploration

---

## Key Technical Details

### State Space

```
State: [theta, alpha, theta_dot, alpha_dot]
- theta: Pendulum angle (rad), 0 = upright
- alpha: Wheel angle (rad)
- theta_dot: Pendulum angular velocity (rad/s)
- alpha_dot: Wheel angular velocity (rad/s)
```

### Physical Parameters (from MATLAB System ID)

```python
g = 9.81        # m/s²
Mh = 0.149      # kg - pendulum mass
Mr = 0.144      # kg - wheel mass
L = 0.14298     # m - pendulum COM to pivot
d = 0.0987      # m
r = 0.1         # m - wheel outer radius
r_in = 0.0911   # m - wheel inner radius

Jh = (1/3) * Mh * L²     # Pendulum inertia
Jr = (1/2) * Mr * (r² + r_in²)  # Wheel inertia

# Damping
lambda = 0.15060423  # From MATLAB lambda.mat
b1 = 0               # Pendulum damping
b2 = 2 * lambda * (Jh + Jr)  # Wheel damping
```

### LQR Gains

```python
# Base gains for inverted pendulum (theta=0 upright)
# Computed via scipy ARE for plant WITH back-EMF (Kv=0.285)
# Q=diag(1, 0, 0.1, 0.001), R=1, voltage-unit B matrix
base_K = [-45.0, 0.0, -5.2, -0.62]

# Applied gains
K = lqr_gain_scale * base_K
# Default: lqr_gain_scale = 1.0 (optimal, used with friction)
```

### Motor Model

```python
# Back-EMF constant (from no-load motor specs)
Kv = 0.285  # V/(rad/s)

# Motor torque (includes back-EMF)
tau_motor = Kt * (V - Kv * alpha_dot) / Rm

# Back-EMF damping: Kt*Kv/Rm = 0.00744 Nm·s/rad (10.6× stronger than b2)
```

### Reward Function

```python
reward = -(theta² + 0.1*theta_dot² + 0.001*alpha_dot² + 0.005*u_RL² + 0.005*(u_RL - u_RL_prev)²)
bonus = +1.0 if |theta| < 0.1 rad
```

Tuning rationale (after back-EMF fix):
- `control_weight=0.005`: low because back-EMF naturally limits wheel speed
- `smoothness_weight=0.005`: mild chatter penalty
- Higher values (0.05/0.02) caused RL to learn to do nothing (cost > benefit)

---

## Files Modified

### `simulation/config.py` (Centralized Configuration)
- All challenge scenarios defined here
- `ChallengeConfig.friction_compensation()` - primary research scenario
- `ChallengeConfig.optimal_lqr_baseline()` - no-friction reference
- `FrictionParams.research_friction()` - Ts=0.15, Tc=0.09, vs=0.02
- Default residual_scale = 4.0V

### `simulation/envs/reaction_wheel_env.py`
- Corrected friction coupling (inverse mass matrix on both equations)
- RK4 integration with 10 sub-steps per control period
- Uses ChallengeConfig for all parameters

### `simulation/train.py`
- Updated for stiction compensation scenario
- Default: optimal LQR (scale=1.0) + Stribeck friction (Ts=0.15)
- Residual scale = 4.0V for stiction compensation

### `simulation/validate.py`
- Three-way comparison:
  1. No-friction LQR (scale=1.0, target: 0.78°)
  2. Friction LQR (scale=1.0, Ts=0.15, problem: ~1.19°)
  3. Hybrid (LQR + RL, solution: <0.9°)
- Phase portraits and friction analysis

### `simulation/tune_friction.py` (NEW)
- Friction parameter sweep tool
- Finds limit cycle / damping regimes
- Generates plots for analysis

---

## Commands

### Training
```bash
# Train with stiction compensation scenario (default config)
python -m simulation.train --timesteps 500000 --save_path models/ppo_residual

# Monitor training
tensorboard --logdir ./logs/
```

### Validation
```bash
# Validate trained model (three-way comparison)
python -m simulation.validate --model_path models/ppo_residual/final_model

# Quick validation (fewer episodes)
python -m simulation.validate --model_path models/ppo_residual/final_model --episodes 10
```

### Friction Analysis
```bash
# Sweep friction parameters around research operating point
python -m simulation.tune_friction --ts_min 0.10 --ts_max 0.20 --n_points 5 --n_episodes 10

# Test specific friction value
python -m simulation.tune_friction --test_single 0.15
```

---

## Lessons Learned

1. **Understand the physics convention** - Sign errors in dynamics equations cause fundamental issues

2. **Sharp transitions are problematic** - If there's no "struggling" regime between success and failure, RL has nothing to learn

3. **Control effort matters** - A solution that requires constant saturation is not a good research contribution

4. **Don't artificially handicap the baseline** - Undertuning LQR or adding random disturbances feels like cheating

5. **Friction can HELP, not hurt** - Counter-intuitive finding that changed our approach

6. **Look for genuine nonlinearities** - Virtual damping is a real problem with a learnable solution

7. **Match training and validation** - Environment parameters must be consistent

---

## Final Results (Stiction Compensation Scenario)

### Validated Results (model: `ppo_bemf_v3`)

| Metric | No-friction LQR | Friction LQR | Hybrid (LQR+RL) |
|--------|-----------------|--------------|------------------|
| **RMS Error** | 0.24° | 0.90° | **0.70°** (22.3% improvement) |
| Episode Length | 500 (100%) | 500 (100%) | 500 (100%) |
| Mean |u_RL| | — | — | 3.5V |
| Mean signed u_RL | — | — | ~0.04V (no bias) |

### Why Stiction Compensation is Compelling

1. **Genuine problem**: Stiction dead zones degrade real control systems
2. **No artificial handicap**: LQR at full optimal gains (scale=1.0)
3. **Clear RL role**: Provide supplemental torque to overcome stiction
4. **Reviewer-proof**: Can't be dismissed with "just use better gains"
5. **Practical**: Real bearings have friction that varies with wear

### What the RL Should Learn

The RL should learn to provide supplemental torque when LQR commands are
insufficient to overcome stiction. The target behavior:
- At small angles (inside dead zone): add torque to break through stiction
- At large angles (outside dead zone): LQR handles it, RL stays quiet
- The net effect: reduce the stiction dead zone from 6.6° to ~2.0°

---

## Next Steps

1. ~~Discover friction helps (not hurts)~~ ✓
2. ~~Develop virtual damping research angle~~ ✓ (deprecated)
3. ~~Fix physics model (friction coupling + RK4)~~ ✓
4. ~~Pivot to stiction compensation with corrected physics~~ ✓
5. ~~Update configs and defaults for friction compensation~~ ✓
6. ~~Fix missing back-EMF (Kv=0 → 0.285)~~ ✓
7. ~~Recompute LQR gains for back-EMF plant~~ ✓
8. ~~Train model with corrected physics (ppo_bemf_v3)~~ ✓ (22.3% improvement)
9. Re-enable domain randomization for sim-to-real robustness
10. Prepare for sim-to-real transfer to ESP32
11. Test with real hardware

---

## References

- MATLAB System ID: `Matlab/modelo_pendulo.m`, `Matlab/lambda.mat`
- Firmware LQR: `firmware/src/main.cpp`
- Environment: `simulation/envs/reaction_wheel_env.py`
