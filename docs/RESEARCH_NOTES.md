# Research Notes: Residual RL for Reaction Wheel Pendulum

## Project Goal

Develop a hybrid control system where:
- **LQR** handles stabilization (the core control task)
- **RL (PPO)** learns a residual policy to provide virtual damping

**Control Law:** `u_total = u_LQR + α * π_θ(s)`

**Current Research Angle:** Virtual Damping
- Underdamped LQR (moderate gains) → oscillatory response
- RL learns `u_RL ∝ -ω` (velocity-proportional damping)
- Hybrid achieves optimal performance without physical friction

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

### Phase 4: Virtual Damping Approach (Current)

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
base_K = [-50.0, 0.0, -6.45, -0.34]

# Applied gains
K = lqr_gain_scale * base_K
# Default: lqr_gain_scale = 0.35 (moderate gains → underdamped)
# Optimal: lqr_gain_scale = 1.0 (well-damped reference)
```

### Reward Function

```python
reward = -(theta² + 0.1*theta_dot² + 0.001*alpha_dot² + 0.01*u_RL²)
bonus = +1.0 if |theta| < 0.1 rad
```

Note: Control cost coefficient (0.01) balances:
- Too low: RL learns bang-bang control
- Too high: RL is too conservative, can't improve

---

## Files Modified

### `simulation/config.py` (NEW - Centralized Configuration)
- All challenge scenarios defined here
- `ChallengeConfig.underdamped_lqr()` - primary research scenario
- `ChallengeConfig.optimal_lqr_baseline()` - reference performance
- `FrictionParams` class for friction experiments

### `simulation/envs/reaction_wheel_env.py`
- Fixed gravity term sign for inverted pendulum
- Updated LQR gains (base_K = [-50, 0, -6.45, -0.34])
- Uses ChallengeConfig for all parameters
- Stribeck friction model (optional)

### `simulation/train.py`
- Updated for virtual damping scenario
- Default: underdamped LQR (scale=0.35), no friction
- Residual scale = 2.0V for damping authority

### `simulation/validate.py`
- Updated to compare:
  1. Optimal LQR (scale=1.0, target performance)
  2. Underdamped LQR (scale=0.35, oscillatory)
  3. Hybrid (LQR + RL virtual damping)
- Phase portraits and damping analysis

### `simulation/tune_friction.py` (NEW)
- Friction parameter sweep tool
- Finds limit cycle / damping regimes
- Generates plots for analysis

---

## Commands

### Training
```bash
# Train with virtual damping scenario (default config)
python -m simulation.train --timesteps 500000 --save_path models/ppo_virtual_damping

# Monitor training
tensorboard --logdir ./logs/
```

### Validation
```bash
# Validate trained model
python -m simulation.validate --model_path models/ppo_virtual_damping/final_model

# Quick validation (fewer episodes)
python -m simulation.validate --model_path models/ppo_virtual_damping/final_model --episodes 10
```

### Friction Analysis
```bash
# Sweep friction parameters to understand damping effects
python -m simulation.tune_friction --ts_min 0.02 --ts_max 0.15 --n_points 8 --plot

# Test specific friction value
python -m simulation.tune_friction --test_single 0.08
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

## Final Results (Virtual Damping Scenario)

### Expected Validation Results

With the virtual damping configuration:

| Metric | Underdamped LQR | Hybrid (LQR+RL) | Target |
|--------|-----------------|-----------------|--------|
| **RMS Error** | ~3° | <1° | Match optimal LQR |
| Episode Length | 500 steps (100%) | 500 steps (100%) | Both survive |
| Oscillations | 6+ peaks | 2-3 peaks | Damped response |

### Reference: Optimal LQR (scale=1.0)
- RMS Error: ~0.9°
- Oscillations: 2-3 peaks
- This is the performance we want the hybrid to match

### Why Virtual Damping is Compelling

1. **Genuine problem**: Underdamped oscillations are common in control
2. **Clear RL role**: Learn damping function `u_RL ∝ -ω`
3. **No cheating**: LQR at moderate (not sabotaged) gains
4. **Interpretable**: Can verify RL learned damping
5. **Practical**: More reliable than physical friction

### What the RL Should Learn

The target function is approximately:
```
u_RL(state) ≈ -k * alpha_dot
```
Where `k` is a learned damping coefficient. This is:
- A simple, linear function of wheel velocity
- Equivalent to viscous friction
- Easy to verify by plotting u_RL vs alpha_dot

---

## Next Steps

1. ~~Discover friction helps (not hurts)~~ ✓
2. ~~Develop virtual damping research angle~~ ✓
3. ~~Update configuration for underdamped LQR~~ ✓
4. Train model with virtual damping scenario
5. Verify RL learns damping function
6. Prepare for sim-to-real transfer to ESP32
7. Test with real hardware

---

## References

- MATLAB System ID: `Matlab/modelo_pendulo.m`, `Matlab/lambda.mat`
- Firmware LQR: `firmware/src/main.cpp`
- Environment: `simulation/envs/reaction_wheel_env.py`
