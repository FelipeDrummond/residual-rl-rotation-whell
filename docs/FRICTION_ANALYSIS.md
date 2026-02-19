# Friction Analysis: Key Experimental Findings

## Summary

Our research journey with friction revealed a surprising finding:

**Friction HELPS the LQR by providing natural damping!**

This document details our friction experiments and the evolution of our research approach:
1. Extreme friction experiments (failed - too sharp transition)
2. Discovery that friction provides beneficial damping
3. Pivot to virtual damping research angle

---

## Stribeck Friction Model

```
τ_f(ω) = [Tc + (Ts - Tc)·exp(-|ω|/vs)]·sign(ω) + σ·ω
```

| Parameter | Symbol | Role |
|-----------|--------|------|
| Static friction | Ts | Stiction at zero velocity |
| Coulomb friction | Tc | Kinetic friction (sliding) |
| Stribeck velocity | vs | Transition width |
| Viscous coefficient | σ | Velocity-proportional damping |

**Standard ratios used:**
- `Tc = 0.6 × Ts`
- `σ = 0.3 × Ts`
- `vs = 0.1 rad/s`

---

## Experiment 1: Finding the LQR Failure Point

### Method
We swept static friction `Ts` from 0 to 0.6 and measured:
- LQR success rate (100 episodes each)
- Control effort (% time at saturation)
- RMS angle error

### Results

| Ts | Success Rate | Saturation | RMS Error |
|----|-------------|------------|-----------|
| 0.00 | 100% | 0% | 0.02° |
| 0.20 | 100% | 0% | 0.03° |
| 0.30 | 100% | 0% | 0.05° |
| 0.40 | 100% | 0% | 0.08° |
| 0.45 | 100% | 0% | 0.12° |
| 0.455 | 100% | 0% | 0.15° |
| 0.456 | 100% | 0% | 0.16° |
| **0.4565** | **72%** | **85%** | **5.2°** |
| **0.457** | **43%** | **97%** | **8.1°** |
| 0.458 | 12% | 100% | - |
| 0.46 | 0% | - | - |
| 0.50 | 0% | - | - |

### Key Finding: The Cliff Edge

```
Success Rate
    100% ─────────────────────╮
                              │
                              │ ← Transition occurs over
                              │   Ts ∈ [0.456, 0.458]
                              │
      0% ──────────────────────────────
           0.3    0.4    0.456  0.5    Ts
```

**The transition from 100% to 0% success occurs over a range of just 0.002 in Ts!**

---

## Experiment 2: Control Effort Analysis

### At the Transition Point (Ts = 0.457)

When LQR barely succeeds:
- **97% of time at saturation** (±12V)
- Control signal oscillates between extremes
- System is on the edge of instability

```
Control Signal (V)
   +12 ─────╮    ╭────╮    ╭────╮    ╭───
            │    │    │    │    │    │
     0 ─────┼────┼────┼────┼────┼────┼───
            │    │    │    │    │    │
   -12 ─────╰────╯    ╰────╯    ╰────╯
            ↑ Bang-bang control
```

### Physical Interpretation

At extreme friction:
1. Stiction prevents wheel from spinning
2. Controller applies max torque to overcome stiction
3. Once moving, friction drops suddenly (Stribeck effect)
4. Controller overshoots, reverses
5. Stiction again at zero velocity
6. Cycle repeats → Limit cycle oscillation

---

## Experiment 3: RL Training with Extreme Friction

### Configuration
```python
friction_params = {
    "Ts": 0.46,   # Just past LQR failure
    "Tc": 0.276,
    "vs": 0.1,
    "sigma": 0.138
}
# Domain randomization: Ts ∈ [0.38, 0.52]
# ~43% baseline LQR success rate
```

### Training Results

| Metric | Value |
|--------|-------|
| Timesteps | 500,000 |
| Episode Length | 28 ± 15 steps (stuck) |
| Episode Reward | -125 ± 40 (poor) |
| Learning Progress | None observed |

**Observation:** Episode length never improved beyond ~30 steps, indicating the RL couldn't learn a useful policy.

### Validation Results (Trained Model)

| Metric | LQR Only | Hybrid (LQR+RL) |
|--------|----------|-----------------|
| Survival | 43% | 74% (+31%) |
| RMS Error | 0.48 rad | 0.34 rad |
| Saturation | 97% | **99%** |

**Problem:** The RL "improvement" came from applying even MORE control effort, not smarter control.

---

## Why Extreme Friction Doesn't Work

### 1. No Gradual Performance Degradation

For RL to learn effectively, we need:
- **Bad:** LQR performs poorly (room to improve)
- **Good:** RL improves performance with reasonable effort

With extreme friction:
- LQR either works perfectly OR fails completely
- No intermediate "struggling but surviving" regime

### 2. Maximum Effort Required

When friction is just above the LQR threshold:
- System is only marginally controllable
- Even perfect control requires maximum effort
- RL can't do better than bang-bang

### 3. Trivial Solution

The RL learns: "Always apply maximum torque in the direction opposite to velocity"
- This is not a learned friction compensation
- This is just harder bang-bang control
- No generalization value

### 4. Physically Unrealistic

Real systems don't have friction that's perfectly at the controllability limit:
- Either the system is clearly controllable (friction is manageable)
- Or it's clearly broken (friction is too high)
- The sharp transition is an artifact of perfect simulation

---

## Key Discovery: Friction Provides Damping!

After extensive experimentation with different LQR gain scales and friction levels, we discovered a counter-intuitive finding:

### Experiment: LQR Gain Scale vs Friction

| LQR Gain Scale | No Friction | With Friction (Ts=0.1) |
|----------------|-------------|------------------------|
| 0.15 | 0% success | 0% success |
| 0.20 | 0% success | 0% success |
| **0.25** | **0% success** | **100%**, 8.7° RMS |
| 0.30 | 100%, 12.7° RMS | 100%, 1.8° RMS |
| 0.35 | 100%, 3.0° RMS | 100%, 1.3° RMS |
| 0.50 | 100%, 1.2° RMS | 100%, 1.1° RMS |
| 1.00 | 100%, 0.9° RMS | 100%, 0.8° RMS |

### Key Insights

1. **At scale=0.25:** LQR fails without friction but succeeds with it!
2. **At scale=0.30:** RMS error drops from 12.7° to 1.8° with friction
3. **At higher scales:** Friction has less impact (LQR already well-damped)

### Physical Explanation

The system without friction is **underdamped**:
- The wheel spins freely with minimal resistance
- LQR commands create oscillations that take time to settle
- Moderate LQR gains can't damp oscillations quickly enough

With friction:
- Friction acts like viscous damping (`τ_f ∝ -ω`)
- This naturally damps oscillations
- Even weak LQR gains can stabilize with friction's help

---

## Current Research Angle: Stiction Compensation

After fixing the physics model (friction coupling on both equations + RK4), we pivot from virtual damping to **stiction compensation**:

**The RL agent learns supplemental torque to overcome stiction dead zones.**

### Physics Model Fix (Phase 5)

The original friction model had two bugs:
1. Missing pendulum reaction (Newton's third law at bearing)
2. Wrong wheel coupling (simple -τ_f/Jr instead of mass matrix)

After fix: friction properly couples through inverse mass matrix M^(-1):
```
theta_ddot += +tau_f / MrL2_Jh
alpha_ddot += -tau_f * (MrL2_Jh + Jr) / (Jr * MrL2_Jh)
```

### Friction Sweep (Corrected Physics, Optimal LQR scale=1.0)

| Ts (Nm) | RMS Error | Survival | Notes |
|---------|-----------|----------|-------|
| 0.00 | 0.78° | 100% | No-friction baseline |
| 0.05 | 0.85° | 100% | Mild degradation |
| 0.10 | 1.02° | 100% | Moderate |
| **0.15** | **1.19°** | **100%** | **Research operating point** |
| 0.20 | 1.45° | 100% | Significant |
| 0.25 | ~2.0° | ~95% | Near failure |

### Stiction Dead Zone Analysis

The dead zone is the angle range where LQR torque < Ts (wheel stuck):

| Ts (Nm) | Dead zone (LQR alone) | Dead zone (LQR + 4V RL) | Reduction |
|---------|-----------------------|--------------------------|-----------|
| 0.10 | θ < 4.4° | θ < 1.3° | 70% |
| **0.15** | **θ < 6.6°** | **θ < 2.0°** | **70%** |
| 0.20 | θ < 8.8° | θ < 2.6° | 70% |

### RL Authority Analysis

How residual_scale affects the dead zone at Ts=0.15:

| residual_scale (V) | Dead zone | Notes |
|--------------------|-----------|-------|
| 0 (LQR only) | θ < 6.6° | Baseline |
| 2 | θ < 4.4° | Some improvement |
| **4** | **θ < 2.0°** | **Chosen: good trade-off** |
| 6 | θ < 0.9° | RL dominates too much |

### Why Ts=0.15 and residual_scale=4V

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Ts | 0.15 Nm | Large enough to degrade LQR (0.78°→1.19°), small enough for 100% survival |
| Tc | 0.09 Nm | Standard 0.6×Ts |
| vs | 0.02 rad/s | Sharp stiction (realistic bearing) |
| sigma | 0.0 | Isolates stiction effect |
| residual_scale | 4V | Reduces dead zone 70% without overwhelming LQR |
| lqr_gain_scale | 1.0 | Full optimal gains (no artificial weakening) |

### Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lqr_gain_scale` | 1.0 | Optimal LQR (reviewer-proof) |
| `friction` | Ts=0.15, Tc=0.09, vs=0.02 | Stiction dead zone |
| `residual_scale` | 4.0 | Overcome stiction |

---

## Lessons for Future Research

1. **Test the baseline thoroughly** - Understand all interactions before assuming the problem

2. **Sharp transitions are problematic** - No gradual degradation means no learning signal

3. **Control effort is a key metric** - Maximum effort solutions are not useful

4. **Fix the physics first** - Friction coupling bugs led to wrong conclusions

5. **Stiction dead zones are genuine** - Real problem in mechanical control systems

6. **Don't weaken the baseline** - Using optimal LQR makes the research reviewer-proof

---

## Data Files and Commands

To reproduce friction experiments:

```bash
# Run friction sweep around research operating point
python -m simulation.tune_friction --ts_min 0.10 --ts_max 0.20 --n_points 5 --n_episodes 10

# Test specific friction value
python -m simulation.tune_friction --test_single 0.15

# Train with stiction compensation (default config)
python -m simulation.train --timesteps 500000 --save_path models/ppo_residual
```
