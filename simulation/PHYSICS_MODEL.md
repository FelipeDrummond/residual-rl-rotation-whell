# Physics Model Documentation

## Summary of Fixes (2024-12-08)

### Critical Bug Fixed: Voltage Scaling
**Problem:** The dynamics equations were applying 12V scaling **twice**, causing the motor to be 12x stronger than intended. This made the system wildly unstable - the pendulum would fall in ~0.17 seconds.

**Root Cause:**
```python
# OLD (WRONG):
tau_motor = (self.Kt / self.Rm) * u_total * 12.0  # u_total is already in Volts!
theta_ddot = ... - (self.Kt / self.Rm) * u_total * 12.0 / MrL2_Jh  # Multiplied by 12 again!
```

**Solution:** Normalized control input to [-1, 1] and applied 12V scaling only in the state-space coefficients (l_3 and l_4), exactly as in MATLAB.

```python
# NEW (CORRECT):
u_normalized = u_total / 12.0  # Normalize voltage to [-1, 1]
l_3 = -(12.0 * self.Kt) / (self.Rm * MrL2_Jh)  # 12V scaling here
theta_ddot = ... + l_3 * u_normalized  # No duplicate scaling
```

## Physics Model Alignment with MATLAB

The Python environment now **exactly matches** the MATLAB `modelo_pendulo.m` state-space model with two extensions:

1. **Non-linear gravity term:** Uses `sin(θ)` instead of linearized `θ`
2. **Stribeck friction:** Added to wheel dynamics for RL compensation research

### State-Space Equations

From MATLAB (lines 36-54), the linear model is:

```
dx/dt = Ax + Bu
where x = [theta, alpha, theta_dot, alpha_dot]'
      u = normalized control input (0 to 1)
```

**A Matrix Elements:**
```
l_31 = -(Mr*g*L + Mh*g*d)/(Mr*L² + Jh)      [Gravity term for pendulum]
l_33 = -b1/(Mr*L² + Jh)                      [Pendulum damping]
l_34 = (b2 + Kt*Kv/Rm)/(Mr*L² + Jh)         [Coupling damping]
l_41 = (Mr*g*L + Mh*g*d)/(Mr*L² + Jh)       [Gravity coupling to wheel]
l_43 = b1/(Mr*L² + Jh)                       [Pendulum-wheel damping coupling]
l_44 = -((Mr*L² + Jh + Jr)*(b2 + Kt*Kv/Rm))/(Jr*(Mr*L² + Jh))  [Wheel damping]
```

**B Matrix Elements:**
```
l_3 = -(12*Kt)/(Rm*(Mr*L² + Jh))                        [Motor input to pendulum]
l_4 = (12*Kt*(Mr*L² + Jh + Jr))/(Rm*Jr*(Mr*L² + Jh))   [Motor input to wheel]
```

### Non-Linear Extension

The Python implementation extends the linear model:

**Pendulum Acceleration (row 3):**
```
θ̈ = l_31*sin(θ) + l_33*θ̇ + l_34*α̇ + l_3*u
```
Note: `sin(θ)` instead of `θ` for non-linear gravity

**Wheel Acceleration (row 4):**
```
α̈ = l_41*sin(θ) + l_43*θ̇ + l_44*α̇ + l_4*u - τ_friction/Jr
```
Note: Added Stribeck friction term `τ_friction/Jr`

## Stribeck Friction Model (RESEARCH)

**Purpose:** Simulates non-linear friction to test residual RL compensation. The real hardware does NOT have this friction - it's added artificially.

**Friction Torque:**
```
τ_friction = [Tc + (Ts - Tc)*exp(-|ω|/vs)] * sign(ω) + σ*ω
```

**Parameters (tunable):**
- Ts = 0.15 Nm (static friction)
- Tc = 0.08 Nm (Coulomb friction)
- vs = 0.05 rad/s (Stribeck velocity)
- σ = 0.02 (viscous friction coefficient)

This creates a "stick-slip" effect that the LQR controller cannot handle, requiring the RL agent to learn compensating torques.

## Motor Parameters

From MATLAB and hardware specs:

- **Stall torque:** τ_stall = 0.3136 Nm @ 1.8A
- **Motor resistance:** Rm = 12V / 1.8A = 6.67 Ω
- **Torque constant:** Kt = 0.3136 / 1.8 = 0.1742 Nm/A
- **Back-EMF constant:** Kv = 0 (simplified model, no back-EMF)
- **Max voltage:** 12V

## Key Differences: MATLAB vs Python

| Property | MATLAB | Python |
|----------|--------|--------|
| Control input | u ∈ [0,1] (normalized) | u ∈ [-12V, +12V], then normalized |
| Gravity term | θ (linear) | sin(θ) (non-linear) |
| Friction | None | Stribeck (on wheel) |
| Integration | lsim (state-space) | Euler with dt=0.02s |
| Back-EMF | Kv = 0 | Kv = 0 |

## Validation

To validate the fixed model, the LQR-only controller should now:
- ✅ Balance the pendulum for multiple seconds (without friction)
- ✅ Achieve episode lengths of 100-500 steps
- ✅ Maintain rewards close to 0 (near upright)

With Stribeck friction enabled:
- ⚠️ LQR may struggle (limit cycles, oscillations)
- ✅ Residual RL should learn to compensate and restore performance

## Expected Training Results

After fixing the physics bugs, training should show:

**LQR-only baseline (no friction):**
- Episode length: 400-500 steps (reaching max_steps)
- Reward: -5 to 0

**Hybrid (LQR + RL with friction):**
- Episode length: Should approach 500 steps as agent learns
- Reward: Should improve from -50 (initial) to -10 or better
- Learning curve: Clear improvement over 200k-500k timesteps

## References

- MATLAB model: `Matlab/modelo_pendulo.m`
- Python implementation: `simulation/envs/reaction_wheel_env.py`
- Firmware LQR: `firmware/src/main.cpp` (line 282)
