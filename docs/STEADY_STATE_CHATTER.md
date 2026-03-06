# Steady-State Chatter Investigation

## The Problem

The hybrid controller (LQR + RL) achieves better overall RMS (0.62° vs 0.91°) than friction LQR alone, but exhibits a qualitative defect: **it never settles to zero in steady state**. Instead, the RL agent continues applying large oscillating voltages (±4V, std=3.86V) even when the pendulum is near upright, causing persistent ~0.5° oscillation.

## Observed Behavior (same initial conditions, seed=12345)

| Metric | Friction LQR | Hybrid (LQR+RL) |
|---|---|---|
| Overall RMS | 0.91° | **0.62°** |
| First 2s mean |θ| (transient) | 1.20° | **0.65°** |
| Last 2s mean |θ| (steady state) | **0.00°** | 0.48° |
| RL mean |u_RL| | — | 3.80V |
| RL std | — | 3.86V |

**The trade-off:** Hybrid wins on transient recovery (overcomes stiction faster) but loses on steady-state precision (chatter prevents settling).

## Why This Happens

### The RL's learned behavior

The RL agent learned a policy that is essentially "always act at near-max authority" regardless of state. The policy std from training converged to a value that, combined with the deterministic mean, produces large actions everywhere in the state space.

### Why the reward function allows this

1. **Upright bonus is always earned**: `+1.0 if |θ| < 0.1 rad (5.7°)` — at 0.5° oscillation, the bonus is always collected
2. **Control penalty is too weak**: `0.005 × u_RL²` — at 4V, this costs only `0.005 × 16 = 0.08` per step, far less than the upright bonus of 1.0
3. **No incentive to improve below 0.5°**: The angle cost `θ²` at 0.5° is `(0.0087)² = 0.000076` — negligible compared to other reward terms
4. **Smoothness penalty is too weak**: `0.005 × Δu²` — with the RL oscillating between similar values each step, the penalty is small

### Why friction LQR settles perfectly

Once LQR drives θ close enough to zero, the commanded voltage drops below the stiction threshold. The wheel locks, and the pendulum sits at the equilibrium point held by stiction. Stiction acts as a **free brake** at steady state — it's a feature, not a bug.

The RL agent disrupts this natural braking by constantly applying voltage that breaks through stiction even when it's not needed.

## Hypotheses for Solutions

### H1: Increase control and smoothness penalties

**Idea:** Make the cost of acting comparable to the cost of the angle error at steady state.

```python
control_weight: float = 0.02   # was 0.005 → 4× increase
smoothness_weight: float = 0.02  # was 0.005 → 4× increase
```

At 4V: control cost = `0.02 × 16 = 0.32/step`. This is still below the upright bonus (1.0) but significant enough that the RL should prefer smaller actions when θ is already small.

**Risk:** May revert to the "RL does nothing" problem from v1 where penalties killed all RL activity. Need to find the sweet spot.

### H2: Angle-gated RL authority

**Idea:** Scale the RL output by a function of |θ| so that RL authority automatically diminishes near the equilibrium.

```python
# In env.step():
gate = min(1.0, |theta| / theta_gate_threshold)
u_RL = gate * residual_scale * action
```

With `theta_gate_threshold = 0.05 rad (2.9°)`:
- At θ = 5°: gate = 1.0 (full authority)
- At θ = 1°: gate = 0.35 (reduced)
- At θ = 0.1°: gate = 0.035 (nearly zero)

**Advantage:** No retraining needed — the gating is deterministic and applied at inference time. The RL still learns to overcome stiction during transients, but physically cannot chatter at steady state.

**Risk:** Introduces a discontinuity in the control law. May reduce performance at intermediate angles.

### H3: Tiered upright bonus

**Idea:** Replace the single threshold bonus with a graduated bonus that rewards precision:

```python
if |theta| < 0.01:    # < 0.57°
    bonus = 2.0
elif |theta| < 0.05:  # < 2.9°
    bonus = 1.5
elif |theta| < 0.1:   # < 5.7°
    bonus = 1.0
```

This creates an incentive to go from 0.5° to 0.1° to 0°. Currently the reward landscape is flat below 5.7°.

**Risk:** May need careful tuning to avoid instability in the reward gradient.

### H4: Separate transient/steady-state rewards

**Idea:** After N steps (e.g., 200 = 4 seconds), increase control penalties significantly:

```python
if step > 200:
    control_weight *= 5  # heavily penalize acting in steady state
```

This lets the RL be aggressive during transient recovery but forces it to shut up at steady state.

**Risk:** Creates a non-stationary reward function which can confuse PPO. May need curriculum learning.

### H5: Action magnitude in observation + penalty schedule

**Idea:** The RL already sees `prev_action` in its observation. Add a penalty term proportional to `|action| × (1 - |theta|/theta_max)` — penalizes large actions when the angle is small.

```python
# Penalty that increases as theta approaches zero
steady_state_penalty = action² × max(0, 1 - |theta| / 0.1)
```

At θ=0: penalty = action² (full penalty for any action)
At θ=0.1 rad: penalty = 0 (no penalty, act freely)

### Recommended approach

**Start with H2 (angle-gated RL)** — it's the simplest, requires no retraining, and directly addresses the root cause. If the gating proves too crude, combine with **H1 (increased penalties)** and retrain.

## Solution Implemented: H2 (Angle-Gated RL) + Retrain

We implemented H2 and retrained the model with the gate built into the environment:

```python
# In env.step(), after computing u_RL:
gate = min(1.0, abs(theta) / theta_gate_threshold)  # threshold = 0.05 rad (2.9°)
u_RL = gate * u_RL
```

### Results (50 episodes, identical seeds)

| Metric | Friction LQR | Hybrid (LQR+RL) |
|---|---|---|
| Overall RMS | 0.90° | **0.48°** (46.8% improvement) |
| First 2s mean \|θ\| (transient) | 1.21° | **0.93°** |
| Last 2s mean \|θ\| (steady state) | 0.000° | **0.003°** |
| Last 2s \|u_RL\| | — | 0.002V |
| Last 2s gate | — | 0.001 |

The chatter is **completely eliminated**. The gate drives RL output to zero at steady state, while the retrained model learned to be effective within the gated authority during transients.

## Validation Fix Applied

The VecNormalize seeding bug was fixed by loading normalization stats manually (pickle) and normalizing observations before `model.predict()`, bypassing the VecNormalize wrapper entirely. Both LQR and hybrid now use the same `env.reset(seed=seed)` path, guaranteeing identical initial conditions.
