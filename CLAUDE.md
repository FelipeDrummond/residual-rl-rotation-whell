# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Residual Reinforcement Learning** project for stiction compensation in a Reaction Wheel Inverted Pendulum. The system uses a hybrid control architecture where an LQR controller handles stabilization, and an RL agent (PPO) provides learned supplemental torque to overcome Stribeck friction (stiction) dead zones.

**Control Law:** `u_total(t) = u_LQR(t) + α · π_θ(s_t)`

**Research Insight:** Stribeck friction creates a stiction dead zone where the motor torque from optimal LQR is insufficient to move the wheel at small pendulum angles. This degrades LQR performance from 0.24° to 0.90° RMS. The RL agent learns supplemental torque to overcome stiction, achieving 0.70° RMS (22.3% improvement).

The workflow is **Sim-to-Real**: train a PPO agent in a Python digital twin (Gymnasium environment) to compensate for stiction, then deploy to ESP32 hardware.

## Repository Structure

```
├── firmware/           # ESP32 embedded C++ code (LQR controller + NN inference)
├── Matlab/            # System identification and LQR tuning scripts
├── simulation/        # Python digital twin (to be implemented)
└── models/            # Trained RL models (ONNX/Zip format)
```

## Firmware (ESP32)

### Building and Uploading

The firmware uses PlatformIO:

```bash
# Build the firmware
cd firmware
pio run

# Upload to ESP32
pio run --target upload

# Monitor serial output (9600 baud)
pio device monitor -b 9600
```

### Hardware Configuration

- **Platform:** ESP32 (fm-devkit board)
- **Control Frequency:** 50Hz (Ts = 0.02s)
- **Encoder Pins:**
  - Wheel: A=17, B=16
  - Pendulum: A=19, B=18
- **Motor Driver:** IN_A=25, IN_B=26 (PWM @ 10kHz)
- **User Input:** Button on GPIO 4 (enables/disables controller)

### Control Architecture

The current implementation runs a pure LQR controller. The state vector is:

**State:** `[theta, alpha, theta_dot, alpha_dot]`
- `theta` (x1): Pendulum angle (rad, adjusted to ±π from vertical)
- `alpha` (x2): Wheel angle (rad)
- `theta_dot` (x3): Pendulum angular velocity
- `alpha_dot` (x4): Wheel angular velocity

**LQR Gains (K):** `[-5.5413, -0.0, -0.7263, -0.0980]`

Control law in [main.cpp:282](firmware/src/main.cpp#L282):
```cpp
u = -K.x1*x.x1 - K.x2*x.x2 - K.x3*x.x3 - K.x4*x.x4
```

**State Estimation:**
- Positions: Read from encoders with conversion factors
  - Wheel: `en2rad_roda = 0.006411413578755`
  - Pendulum: `en2rad_pend = 0.001570796326795`
- Velocities: Backward difference (unfiltered): `(q - q_ant) / Ts`

**Dual-Core FreeRTOS Tasks:**
- `taskControle` (Core 1): Timer-driven at 50Hz, reads sensors, computes control, drives motor
- `taskComunicacao` (Core 0): Sends state telemetry via Serial at 10Hz

### Adding RL Residual

When implementing the residual RL component:
1. The RL policy output should be added to `u` in [main.cpp:228](firmware/src/main.cpp#L228) before saturation
2. Neural network inference code will need to be lightweight (consider fixed-point MLP)
3. Model weights should be stored as C arrays in a header file

## MATLAB System Identification

The [modelo_pendulo.m](Matlab/modelo_pendulo.m) file contains the physical parameters derived from system identification:

**Physical Parameters:**
- Gravity: 9.81 m/s²
- Pendulum mass (Mh): 0.149 kg
- Wheel mass (Mr): 0.144 kg
- Pendulum length (L): 0.14298 m (COM to pivot)
- Distance (d): 0.0987 m
- Wheel outer radius: 0.1 m, inner radius: 0.0911 m
- Motor stall torque: 0.3136 Nm @ 1.8A
- Max voltage: 12V

**Derived Inertias:**
- `Jh = (1/3) * Mh * L²` (pendulum)
- `Jr = (1/2) * Mr * (r² + r_in²)` (wheel)

These parameters MUST be used when creating the Python simulation environment.

## Python Simulation

The digital twin is implemented in `simulation/` with the following structure:

```
simulation/
├── __init__.py
├── envs/
│   ├── __init__.py
│   └── reaction_wheel_env.py    # ReactionWheelEnv (Gymnasium)
├── train.py                      # PPO training script
├── validate.py                   # LQR vs Hybrid comparison
└── README.md
```

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train a PPO agent (auto-detects MPS/CUDA/CPU):
```bash
python -m simulation.train --timesteps 500000 --n_envs 4
```

**GPU Acceleration:**
- Automatically uses MPS on Apple Silicon (M1/M2/M3)
- Automatically uses CUDA on NVIDIA GPUs
- Falls back to CPU if no GPU available

Force specific device:
```bash
python -m simulation.train --device mps   # Apple Silicon
python -m simulation.train --device cuda  # NVIDIA GPU
python -m simulation.train --cpu          # Force CPU
```

Key arguments:
- `--timesteps`: Total training steps (default: 1,000,000)
- `--n_envs`: Parallel environments (default: 4)
- `--residual_scale`: Scaling factor α for residual (default: 4.0)
- `--no_domain_rand`: Disable domain randomization
- `--device`: Device selection (auto/cuda/mps/cpu)

Monitor training:
```bash
tensorboard --logdir ./logs/
```

### Validation

Compare LQR vs Hybrid control:
```bash
python -m simulation.validate --model_path models/ppo_residual/final_model
```

Generates comparison plots and prints metrics (RMS error, reward, episode length).

### Environment Details (`ReactionWheelEnv`)

**Observation Space:** `[theta, alpha, theta_dot, alpha_dot]`
- Uses same physical parameters from MATLAB system ID
- State limits: theta ∈ [-π, π], velocities capped for numerical stability

**Action Space:** Residual control signal (scalar, normalized to [-1, 1])
- Scaled by `residual_scale` parameter before adding to LQR

**Control Architecture (in step function):**
```python
u_LQR = -K · state  # K = [45.0, 0.0, 5.2, 0.62] (computed via scipy ARE with back-EMF)
u_total = u_LQR + residual_scale * action
u_total = clip(u_total, -12V, +12V)
```

**Physics Implementation:**
- Non-linear equations of motion for coupled pendulum-wheel system
- **Motor model with back-EMF**: `tau = Kt*(V - Kv*ω)/Rm`, Kv=0.285 V/(rad/s)
  - Back-EMF damping `Kt*Kv/Rm = 0.00744` is 10.6× stronger than linear damping b2
- Linear damping from MATLAB system ID:
  - `b1 = 0` (no pendulum damping)
  - `b2 = 2*λ*(Jh+Jr) ≈ 0.000703 N⋅m⋅s/rad` (wheel damping, λ=0.15060423)
- **Friction Configuration** (RESEARCH):
  - Default: **Stribeck friction** (Ts=0.08, Tc=0.048, vs=0.02, sigma=0.0)
  - Creates stiction dead zone where motor torque < Ts → wheel stuck
  - **Research scenario (scale=1.0 LQR gains):**
    - No friction: 0.24° RMS (target performance)
    - With friction (Ts=0.08): 0.90° RMS (degraded by stiction)
    - Hybrid (LQR+RL, 4V): **0.70° RMS** (22.3% improvement)
  - **Research angle:** RL learns supplemental torque to overcome stiction
- Initial conditions: theta ∈ [-0.1, 0.1] rad, theta_dot ∈ [-0.2, 0.2] rad/s
- RK4 integration with 10 sub-steps per control period at dt=0.02s

**Reward Function:**
```python
reward = -(theta² + 0.1*theta_dot² + 0.001*alpha_dot² + 0.005*u_RL² + 0.005*(u_RL - u_RL_prev)²)
bonus = +1.0 if |theta| < 0.1 rad
```
- `control_weight=0.005`: low because back-EMF naturally limits wheel speed
- `smoothness_weight=0.005`: mild chatter penalty

**Domain Randomization:** When enabled, randomizes masses, lengths, and friction parameters by ±10% for robust sim-to-real transfer.

**Termination:** Episode ends if |theta| > π/3 or |theta_dot| > 15 rad/s

## Key Implementation Notes

1. **Angle Wrapping:** Pendulum angle is adjusted to range [-π, π] with 0 being upright (see [main.cpp:274](firmware/src/main.cpp#L274))
2. **Voltage Saturation:** Control signal is clamped to ±12V (see [main.cpp:230](firmware/src/main.cpp#L230))
3. **Sign Function:** Custom implementation handles zero case (see [main.cpp:270](firmware/src/main.cpp#L270))
4. **Communication Protocol:** Binary frames with 0xFF header, 5 doubles (x1,x2,x3,x4,u), 0x00 footer

## Research Context

This work is for a research paper: **"Hybrid Control for Reaction Wheel Pendulums: Stiction Compensation via Residual Reinforcement Learning"**

**The Core Problem:**
Stribeck friction creates a stiction dead zone at low wheel velocities where the motor torque commanded by LQR is insufficient to move the wheel. At small pendulum angles, LQR commands small corrections that fall below the stiction threshold — the wheel remains stuck, the pendulum drifts, and performance degrades. Even optimal LQR (scale=1.0) is affected: RMS error increases from 0.78° to ~1.19°.

**The Solution:**
Train an RL agent to provide supplemental torque that helps LQR overcome stiction. The RL adds extra torque when the LQR command alone would be absorbed by stiction. This is:
1. Complementary to LQR (LQR stabilizes, RL overcomes stiction)
2. Targeted (only activates when stiction is the bottleneck)
3. Deployable to embedded systems (small network, 4V authority)

**Results (with angle-gated RL, 50 episodes):**
- No-friction LQR: 0.22° RMS (target performance)
- Friction LQR alone: 0.90° RMS (degraded by stiction)
- Hybrid (LQR + RL): **0.48° RMS** (46.8% improvement, closes 62% of the gap)
- Steady-state chatter: **eliminated** (gate → 0, |u_RL| → 0.002V in last 2s)

**Why This is Compelling:**
1. Genuine problem (stiction dead zones degrade real control systems)
2. LQR is essential (handles the hard stabilization task)
3. RL role is clear (overcomes stiction during transients, fades at equilibrium)
4. Practical benefit (stiction varies in real hardware; RL adapts)
5. Angle-gated RL ensures zero steady-state interference (stiction acts as free brake)
