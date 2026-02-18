# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Residual Reinforcement Learning** project for virtual damping in a Reaction Wheel Inverted Pendulum. The system uses a hybrid control architecture where an LQR controller handles stabilization, and an RL agent (PPO) provides learned virtual damping to reduce oscillations.

**Control Law:** `u_total(t) = u_LQR(t) + α · π_θ(s_t)`

**Research Insight:** The underdamped LQR (moderate gains) stabilizes the pendulum but exhibits significant oscillations. Physical friction helps by providing natural damping, but is unreliable. The RL agent learns virtual damping (`u_RL ∝ -ω`) that is controllable and predictable.

The workflow is **Sim-to-Real**: train a PPO agent in a Python digital twin (Gymnasium environment) to provide virtual damping, then deploy to ESP32 hardware.

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
- `--timesteps`: Total training steps (default: 500,000)
- `--n_envs`: Parallel environments (default: 4)
- `--residual_scale`: Scaling factor α for residual (default: 1.0)
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
u_LQR = -K · state  # Computed internally with K = [5.5413, 0.0, 0.7263, 0.0980]
u_total = u_LQR + residual_scale * action
u_total = clip(u_total, -12V, +12V)
```

**Physics Implementation:**
- Non-linear equations of motion for coupled pendulum-wheel system
- Linear damping from MATLAB system ID:
  - `b1 = 0` (no pendulum damping)
  - `b2 = 2*λ*(Jh+Jr) ≈ 0.000703 N⋅m⋅s/rad` (wheel damping, λ=0.15060423)
- **Damping Configuration** (RESEARCH):
  - Default: **NO FRICTION** - creates underdamped system for RL training
  - Optional Stribeck model available for comparison studies
  - **Research finding:** Friction HELPS LQR by providing damping!
  - **Research scenario (scale=0.35 LQR gains):**
    - Without friction: 100% success, ~3° RMS (oscillatory)
    - With friction: 100% success, ~1.3° RMS (damped)
    - Optimal LQR (scale=1.0): ~0.9° RMS (target)
  - **Research angle:** RL learns virtual damping (`u_RL ∝ -ω`) to match optimal LQR
- Initial conditions: theta ∈ [-0.3, 0.3] rad, theta_dot ∈ [-0.5, 0.5] rad/s
- Euler integration at dt=0.02s

**Reward Function (MRAC-based):**
```python
reward = -(theta² + 0.1*theta_dot² + 0.001*alpha_dot² + 0.01*u_RL²)
bonus = +1.0 if |theta| < 0.1 rad
```

**Domain Randomization:** When enabled, randomizes masses, lengths, and friction parameters by ±10% for robust sim-to-real transfer.

**Termination:** Episode ends if |theta| > π/3 or |theta_dot| > 15 rad/s

## Key Implementation Notes

1. **Angle Wrapping:** Pendulum angle is adjusted to range [-π, π] with 0 being upright (see [main.cpp:274](firmware/src/main.cpp#L274))
2. **Voltage Saturation:** Control signal is clamped to ±12V (see [main.cpp:230](firmware/src/main.cpp#L230))
3. **Sign Function:** Custom implementation handles zero case (see [main.cpp:270](firmware/src/main.cpp#L270))
4. **Communication Protocol:** Binary frames with 0xFF header, 5 doubles (x1,x2,x3,x4,u), 0x00 footer

## Research Context

This work is for a research paper: **"Hybrid Control for Reaction Wheel Pendulums: Virtual Damping via Residual Reinforcement Learning"**

**The Core Problem:**
LQR controllers with moderate gains create underdamped systems with oscillatory transient response. Physical friction can help provide damping, but it's unreliable (varies with temperature, wear, etc.) and not precisely controllable.

**The Solution:**
Train an RL agent to provide virtual damping: `u_RL(ω) ∝ -ω`. This is:
1. Controllable and predictable (unlike physical friction)
2. A simple, interpretable function (easy to verify and deploy)
3. Complementary to LQR (LQR stabilizes, RL damps)

**Expected Results:**
- Underdamped LQR alone: 100% survival, ~3° RMS (oscillatory)
- Hybrid (LQR + RL): 100% survival, <1° RMS (matches optimal LQR)

**Why This is Compelling:**
1. Genuine problem (underdamped oscillations are common in control)
2. LQR is essential (handles the hard stabilization task)
3. RL role is clear (learns damping, not full control)
4. Practical benefit (more reliable than mechanical friction)
