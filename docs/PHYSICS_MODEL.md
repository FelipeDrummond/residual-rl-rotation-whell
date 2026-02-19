# Physics Model: Reaction Wheel Inverted Pendulum

## System Overview

The reaction wheel inverted pendulum consists of:
1. **Pendulum body** - A rod pivoting about a fixed point
2. **Reaction wheel** - A flywheel mounted at the end of the pendulum
3. **DC Motor** - Drives the reaction wheel

The system is underactuated: we can only apply torque to the wheel, which creates a reaction torque on the pendulum.

```
        Pivot Point
            ●
           /|
          / |
         /  |  Pendulum (mass Mh, length L)
        /   |
       /    |
      ● ────┼──── Wheel (mass Mr, radius r)
            |
         Motor

    theta = 0 at vertical (upright)
    theta > 0 clockwise
```

---

## State Variables

| Variable | Symbol | Description | Units |
|----------|--------|-------------|-------|
| Pendulum angle | θ (theta) | Angle from vertical | rad |
| Wheel angle | α (alpha) | Wheel rotation | rad |
| Pendulum velocity | θ̇ (theta_dot) | Angular velocity | rad/s |
| Wheel velocity | α̇ (alpha_dot) | Angular velocity | rad/s |

**Convention:** `theta = 0` is the upright (inverted) position.

---

## Physical Parameters

From MATLAB system identification (`Matlab/modelo_pendulo.m`):

### Masses and Dimensions
```
g   = 9.81 m/s²        # Gravity
Mh  = 0.149 kg         # Pendulum mass
Mr  = 0.144 kg         # Wheel mass
L   = 0.14298 m        # Pendulum length (COM to pivot)
d   = 0.0987 m         # Distance parameter
r   = 0.1 m            # Wheel outer radius
r_in = 0.0911 m        # Wheel inner radius
```

### Inertias
```
Jh = (1/3) * Mh * L²              # Pendulum moment of inertia
Jh ≈ 1.015e-3 kg·m²

Jr = (1/2) * Mr * (r² + r_in²)    # Wheel moment of inertia (hollow cylinder)
Jr ≈ 1.317e-3 kg·m²
```

### Motor Parameters
```
Rm = 6.67 Ω            # Motor resistance (12V / 1.8A stall)
Kt = 0.1742 N·m/A      # Torque constant (τ_stall / i_stall)
Kv = 0.285 V/(rad/s)   # Back-EMF constant (from no-load specs)
u_max = 12 V           # Maximum voltage
w_noload = 380 RPM     # No-load speed
i_noload = 0.1 A       # No-load current
```

**Back-EMF derivation:**
```
Kv = (V - I_noload × Rm) / ω_noload
   = (12 - 0.1 × 6.667) / (380 × 2π/60)
   = 11.33 / 39.8 ≈ 0.285 V/(rad/s)
```

Back-EMF provides velocity-dependent braking: as wheel speed increases, back-EMF
opposes driving voltage → less current → less torque → natural speed limit.
The back-EMF damping term `Kt·Kv/Rm ≈ 0.00744 Nm·s/rad` is **10.6× stronger**
than the linear damping `b2 ≈ 0.000703 Nm·s/rad`.

**Note:** The MATLAB model had `Kt=0, Kv=0` (making B=0, uncontrollable).
The LQR gains in the firmware were computed separately, not from that file.

### Damping
```
λ = 0.15060423         # Damping coefficient (from lambda.mat)
b1 = 0                 # Pendulum damping (none)
b2 = 2 * λ * (Jh + Jr) # Wheel damping
b2 ≈ 7.03e-4 N·m·s/rad
```

---

## Equations of Motion

### Linearized State-Space Form

The system can be written in state-space form:
```
ẋ = A·x + B·u
y = C·x + D·u
```

Where `x = [θ, α, θ̇, α̇]ᵀ` and `u` is the normalized voltage input.

### A Matrix (System Dynamics)

```
A = [  0      0     1     0   ]
    [  0      0     0     1   ]
    [ l_31    0    l_33  l_34 ]
    [ l_41    0    l_43  l_44 ]
```

### Matrix Elements

**Common terms:**
```
MrL²_Jh = Mr·L² + Jh       # Combined inertia term
MhgL_MrgL = Mh·g·d + Mr·g·L  # Gravity moment
```

**Pendulum dynamics (row 3):**
```
l_31 = +MhgL_MrgL / MrL²_Jh     # POSITIVE for inverted pendulum!
l_33 = -b1 / MrL²_Jh
l_34 = (b2 + Kt·Kv/Rm) / MrL²_Jh
```

**Wheel dynamics (row 4):**
```
l_41 = -MhgL_MrgL / MrL²_Jh     # Reaction to pendulum
l_43 = b1 / MrL²_Jh
l_44 = -((MrL²_Jh + Jr)·(b2 + Kt·Kv/Rm)) / (Jr·MrL²_Jh)
```

### B Matrix (Input)

```
B = [   0   ]
    [   0   ]
    [  l_3  ]
    [  l_4  ]
```

Where:
```
l_3 = -(12·Kt) / (Rm·MrL²_Jh)
l_4 = (12·Kt·(MrL²_Jh + Jr)) / (Rm·Jr·MrL²_Jh)
```

**Note:** The factor of 12 converts normalized input [-1, 1] to voltage [-12V, +12V].

---

## Sign Convention (Critical!)

### The Gravity Term Problem

The sign of `l_31` depends on where `theta = 0` is defined:

| Convention | theta=0 at | Gravity effect | l_31 sign |
|------------|-----------|----------------|-----------|
| MATLAB | Bottom (hanging) | Stabilizing | Negative |
| Simulation | Top (inverted) | Destabilizing | **Positive** |

**This was a key bug fix in our research.**

### Physical Interpretation

For an **inverted pendulum** (theta=0 upright):
- Small positive theta → pendulum falls clockwise
- Gravity **accelerates** the fall (destabilizing)
- Therefore `θ̈ ∝ +θ` → `l_31 > 0`

For a **regular pendulum** (theta=0 hanging):
- Small positive theta → pendulum swings back
- Gravity **restores** to equilibrium (stabilizing)
- Therefore `θ̈ ∝ -θ` → `l_31 < 0`

---

## Stribeck Friction Model

### Friction Torque

The friction torque on the wheel is:
```
τ_f(ω) = F_stribeck(ω) + F_viscous(ω)
```

**Stribeck friction:**
```
F_stribeck = [Tc + (Ts - Tc)·exp(-|ω|/vs)]·sign(ω)
```

**Viscous friction:**
```
F_viscous = σ·ω
```

### Parameters

| Parameter | Symbol | Description | Research Value |
|-----------|--------|-------------|----------------|
| Static friction | Ts | Maximum stiction | 0.15 N·m |
| Coulomb friction | Tc | Kinetic friction | 0.09 N·m (0.6·Ts) |
| Stribeck velocity | vs | Transition velocity | 0.02 rad/s |
| Viscous coefficient | σ | Velocity-dependent | 0.0 (isolates stiction) |

### Friction Curve

```
Torque
   ↑
Ts ┤     ╭──────────────
   │    ╱
Tc ┤   ╱
   │  ╱
   │ ╱
───┼╱─────────────────→ ω
   │╲
   │ ╲
-Tc┤   ╲
   │    ╲
-Ts┤     ╰──────────────
```

### Friction Coupling Fix (Phase 5)

**The Bug:** The original implementation only applied friction to the wheel equation as `-τ_f/Jr`, ignoring:
1. Newton's third law: bearing friction creates a reaction torque on the pendulum
2. The coupled dynamics: the mass matrix means friction on the wheel affects both bodies

**The Derivation:** The generalized friction force vector is `Q_f = [+τ_f, -τ_f]` (positive reaction on pendulum, negative on wheel). After applying the inverse mass matrix M^(-1):

```
theta_ddot += +tau_f / MrL2_Jh
alpha_ddot += -tau_f * (MrL2_Jh + Jr) / (Jr * MrL2_Jh)
```

Where `MrL2_Jh = Mr·L² + Jh` is the combined pendulum inertia term.

**Physical Verification:** When friction exactly cancels motor torque (stiction), both coupling terms reduce the motor's effect to zero on BOTH equations. This is physically correct: if the wheel is stuck by friction, the motor cannot exert any torque on the pendulum.

**Impact:** With the corrected coupling, Stribeck friction (Ts=0.15 Nm) degrades optimal LQR from 0.78° to ~1.19° RMS. The old model underestimated friction's effect on the pendulum.

---

## Integration Method

### RK4 with Sub-stepping

Stribeck friction creates stiff dynamics due to the sharp transition at low velocities (vs=0.02 rad/s). Simple Euler integration with dt=0.02s is unstable.

The simulation uses 4th-order Runge-Kutta (RK4) with 10 sub-steps per control period:
```
sub_dt = dt / 10 = 0.002s
for each sub-step:
    k1 = f(x, u)
    k2 = f(x + 0.5·sub_dt·k1, u)
    k3 = f(x + 0.5·sub_dt·k2, u)
    k4 = f(x + sub_dt·k3, u)
    x = x + (sub_dt/6)·(k1 + 2·k2 + 2·k3 + k4)
```

The control input `u` is held constant across all sub-steps (zero-order hold), matching the 50 Hz control frequency of the real hardware.

---

## LQR Controller Design

### Optimal LQR Gains

Computed via scipy ARE for the plant **with back-EMF** (Kv=0.285):
```
K = [-45.0, 0.0, -5.2, -0.62]
```

Q = diag(1, 0, 0.1, 0.001), R = 1, voltage-unit B matrix.

Control law:
```
u_LQR = -K·x = -K₁·θ - K₂·α - K₃·θ̇ - K₄·α̇
```

With these gains:
- K₁ = -45.0: Strong angle feedback (main stabilization)
- K₂ = 0.0: No wheel angle feedback (standard for reaction wheels)
- K₃ = -5.2: Velocity damping (prevents overshoot)
- K₄ = -0.62: Wheel velocity feedback (back-EMF provides most damping)

### Research Configuration (Stiction Compensation)

For the **stiction compensation** scenario:
```
K = 1.0 × K_optimal = [-45.0, 0.0, -5.2, -0.62]
```

With Stribeck friction (Ts=0.15 Nm), optimal LQR degrades from 0.24° to 0.90° RMS.
The RL agent with 4V authority compensates stiction to 0.70° RMS (22% improvement).

---

## Numerical Values (Computed)

With the physical parameters above:

```
MrL²_Jh = 0.144 × 0.14298² + 1.015e-3 = 3.96e-3 kg·m²
MhgL_MrgL = 0.149 × 9.81 × 0.0987 + 0.144 × 9.81 × 0.14298 = 0.346 N·m

l_31 = +0.346 / 3.96e-3 = +87.4 rad/s²
l_33 = 0  (b1 = 0)
l_34 = (7.03e-4 + 0.1742×0.285/6.667) / 3.96e-3 = 2.06 rad/s  (back-EMF dominant)

l_41 = -87.4 rad/s²
l_43 = 0
l_44 = -((3.96e-3 + 1.317e-3) × (7.03e-4 + 0.00744)) / (1.317e-3 × 3.96e-3) = -8.24 rad/s

l_3 = -(12 × 0.1742) / (6.67 × 3.96e-3) = -79.2 V⁻¹
l_4 = (12 × 0.1742 × (3.96e-3 + 1.317e-3)) / (6.67 × 1.317e-3 × 3.96e-3) = 179.3 V⁻¹
```

