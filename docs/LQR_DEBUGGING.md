# LQR Debugging: Sign Conventions and Gain Computation

## The Original Problem

When running the LQR controller with **no friction**, we expected perfect stabilization. Instead, we observed continuous oscillation that never settled.

```
Theta (rad)
   +0.3 ─╮    ╭╮    ╭╮    ╭╮    ╭╮
         │   ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
     0  ─┼──╱────╲╱────╲╱────╲╱────╲──
         │ ╱
   -0.3 ─╯
         0    2    4    6    8   10  time (s)
```

This indicated a fundamental problem with the dynamics or controller.

---

## Root Cause Analysis

### 1. MATLAB Model Analysis

The MATLAB file `modelo_pendulo.m` defines:

```matlab
% State-space matrix elements (line 36-46)
l_31 = -(Mr*g*L + Mh*g*d)/(Mr*L^2 + Jh);  % NEGATIVE
...
A = [0     0    1    0;
     0     0    0    1;
     l_31  0    l_33 l_34;
     l_41  0    l_43 l_44];
```

**Key observation:** `l_31` is **negative**, meaning:
- Positive θ → negative θ̈
- Gravity pulls the pendulum back toward θ=0
- This is **stabilizing** behavior

### 2. The Convention Problem

**MATLAB convention:**
- `theta = 0` at the **bottom** (hanging equilibrium)
- This is a **stable** equilibrium
- Gravity is **restoring** (stabilizing)

**Our simulation convention:**
- `theta = 0` at the **top** (inverted/upright)
- This is an **unstable** equilibrium
- Gravity is **destabilizing**

### 3. Physical Picture

```
MATLAB Model:                Our Model:
                                  ↑ theta = 0
     theta = 0                    │
         │                       ╱│╲
         ▼                     ╱  │  ╲
        ╱│╲                   ●   │
       ╱ │ ╲                      │
      ●  │                        │
         │                        │
  Stable (hanging)         Unstable (inverted)
  Gravity restores         Gravity destabilizes
```

---

## The Fix

### Inverting the Gravity Term

For an **inverted pendulum**, the gravity term must be **positive** (destabilizing):

```python
# BEFORE (wrong - copied from MATLAB)
l_31 = -MhgL_MrgL / MrL2_Jh  # Negative = stabilizing

# AFTER (correct for inverted pendulum)
l_31 = +MhgL_MrgL / MrL2_Jh  # Positive = destabilizing
```

Similarly, the wheel coupling term:
```python
# BEFORE
l_41 = +MhgL_MrgL / MrL2_Jh

# AFTER
l_41 = -MhgL_MrgL / MrL2_Jh
```

### Why These Signs?

**Pendulum equation (θ̈):**
- When θ > 0 (tilted right), gravity pulls it further right
- Therefore θ̈ ∝ +θ (positive coefficient)
- `l_31 > 0`

**Wheel equation (α̈):**
- When pendulum falls right, the reaction torque pushes wheel left
- This is opposite to the pendulum acceleration
- `l_41 = -l_31`

---

## LQR Gain Computation

### Original Firmware Gains

The ESP32 firmware uses:
```cpp
K = {-5.5413, -0.0, -0.7263, -0.0980};
```

These gains were ~10x too small for the corrected dynamics.

### Computing New Gains

Using the corrected A and B matrices, we solve the LQR problem:
```
minimize ∫(x'Qx + u'Ru)dt
```

With typical weights:
```python
Q = diag([10, 0.01, 1, 0.01])  # State weights
R = [1]                         # Control weight
```

**Resulting gains:**
```python
K = [-50.0, 0.0, -6.45, -0.34]
```

### Gain Interpretation

| Gain | Value | Meaning |
|------|-------|---------|
| K₁ | -50.0 | Strong angle correction (main stabilization) |
| K₂ | 0.0 | No wheel position control (can spin freely) |
| K₃ | -6.45 | Velocity damping (prevents overshoot) |
| K₄ | -0.34 | Wheel velocity damping (smooth control) |

### Verification

With corrected dynamics and new gains:
- **No friction, optimal gains (scale=1.0):** 100% success, ~0.9° RMS error
- **No friction, moderate gains (scale=0.35):** 100% success, ~3° RMS error (underdamped)
- **With friction (Ts=0.1):** 100% success, improved RMS (friction provides damping)

**Key finding:** Friction HELPS by providing natural damping. The virtual damping research angle uses moderate gains (scale=0.35) without friction, where RL learns to provide the missing damping.

---

## Debugging Checklist

If you encounter LQR instability, check:

### 1. Sign Conventions
- [ ] Is theta=0 at upright or hanging position?
- [ ] Does l_31 have correct sign for your convention?
- [ ] Does l_41 have opposite sign to l_31?

### 2. LQR Gains
- [ ] Were gains computed with correct A, B matrices?
- [ ] Are gains appropriate magnitude for the system?
- [ ] Sign convention: u = -Kx (negative feedback)

### 3. Control Law
- [ ] Is control applied as u = -K·x (not +K·x)?
- [ ] Is voltage saturation applied correctly?
- [ ] Is angle wrapping to [-π, π] working?

### 4. Physical Parameters
- [ ] Are masses in kg (not grams)?
- [ ] Are lengths in meters (not cm)?
- [ ] Are inertias computed correctly?

---

## Code Locations

### Dynamics (sign-corrected)
File: `simulation/envs/reaction_wheel_env.py`
Function: `_dynamics()`
Lines: 266-287

```python
# Pendulum equation coefficients
l_31 = +MhgL_MrgL / MrL2_Jh  # POSITIVE for inverted pendulum
l_33 = -self.b1 / MrL2_Jh
l_34 = (self.b2 + self.Kt * self.Kv / self.Rm) / MrL2_Jh

# Wheel equation coefficients
l_41 = -MhgL_MrgL / MrL2_Jh  # NEGATIVE (reaction)
```

### LQR Gains
File: `simulation/envs/reaction_wheel_env.py`
Lines: 115-116

```python
base_K = np.array([-50.0, 0.0, -6.45, -0.34])
self.K = self.lqr_gain_scale * base_K
```

### Control Application
File: `simulation/envs/reaction_wheel_env.py`
Function: `_lqr_control()`
Lines: 213-231

```python
def _lqr_control(self, state: np.ndarray) -> float:
    return -np.dot(self.K, state)  # Note: negative sign
```

---

## Comparison: MATLAB vs Python

| Aspect | MATLAB | Python |
|--------|--------|--------|
| theta=0 | Bottom (stable) | Top (unstable) |
| l_31 sign | Negative | **Positive** |
| Kt, Kv | Set to 0 | Set to 0 |
| B matrix | Via motor torque | Via motor torque |
| Control | Not implemented | LQR + RL |

---

## Future Considerations

### If Deploying to Hardware

The firmware (`main.cpp`) may use the original convention. Before deployment:
1. Verify firmware theta convention
2. Adjust gains if needed
3. Test with real hardware gradually

### If Changing Parameters

When modifying physical parameters:
1. Recompute inertias (Jh, Jr)
2. Recompute LQR gains
3. Verify with no-friction simulation
