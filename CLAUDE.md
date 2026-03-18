# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Residual Reinforcement Learning** for cogging torque compensation in a Reaction Wheel Inverted Pendulum. Hybrid control: LQR stabilizes, PPO provides learned supplemental torque for position-dependent cogging that LQR structurally cannot handle (K[1]=0 means no wheel angle feedback).

**Control Law:** `u_total(t) = u_LQR(t) + α · π_θ(s_t)`

**Workflow:** Sim-to-Real — train PPO in Python digital twin (Gymnasium), deploy to ESP32 hardware.

**Paper:** "Hybrid Control for Reaction Wheel Pendulums: Cogging Torque Compensation via Residual Reinforcement Learning"

## Commands

### Python Simulation

```bash
# Install dependencies (use venv, system python has no packages)
pip install -r requirements.txt

# Train PPO agent (auto-detects MPS/CUDA/CPU)
python -m simulation.train --timesteps 500000 --n_envs 4
python -m simulation.train --device mps        # Force Apple Silicon
python -m simulation.train --no_back_emf       # Kv=0 plant variant

# Validate LQR vs Hybrid control (generates comparison plots + metrics)
python -m simulation.validate --model_path models/ppo_residual/final_model

# Smoke tests
python -m simulation.test_env
python -m simulation.test_lqr_only

# Monitor training
tensorboard --logdir ./logs/
```

### Firmware (ESP32, PlatformIO)

```bash
cd firmware
pio run                        # Build
pio run --target upload        # Upload
pio device monitor -b 9600     # Serial monitor
```

## Architecture

### Control Flow

```
ReactionWheelEnv.step():
  u_LQR = -K · state        # K = [-45.0, 0.0, -5.2, -0.62] (scipy ARE, Kv=0.285)
  u_RL = residual_scale * action   # action ∈ [-1, 1], scale = 2.0V
  u_total = clip(u_LQR + u_RL, ±12V)
  → RK4 integration (10 sub-steps per dt=0.02s)
  → cogging torque: τ_cog = 0.05·sin(7·α)
```

### Key Dataflow

- `simulation/config.py` — Single source of truth for all parameters: `PhysicalParams`, `CoggingParams`, `LQRParams`, `ChallengeConfig`, `TrainingConfig`, `EnvConfig`, `RewardConfig`. Also contains `compute_lqr_gains()` which solves the continuous ARE.
- `simulation/envs/reaction_wheel_env.py` — Gymnasium env with non-linear EOM, RK4 integrator, motor model with back-EMF (`tau = Kt*(V - Kv*ω)/Rm`), and internal LQR. The RL agent only controls the residual.
- `simulation/train.py` — PPO training with `VecNormalize`, checkpoints, eval callback, TensorBoard logging.
- `simulation/validate.py` — Three-way comparison (no-cogging LQR / cogging LQR / hybrid). Manual VecNormalize handling (pickle load + manual clip) due to seeding bug.
- `firmware/src/main.cpp` — Dual-core FreeRTOS: `taskControle` (Core 1, 50Hz) and `taskComunicacao` (Core 0, 10Hz). RL residual gets added to `u` before voltage saturation.

### Physics: What Matters

- **Back-EMF is dominant:** `Kt*Kv/Rm = 0.00744` is 10.6× stronger than linear damping b2. Always use Kv=0.285 unless explicitly doing no-back-EMF experiments.
- **Cogging is position-dependent:** `τ_cog = A·sin(N·α)`. LQR with K[1]=0 is structurally blind to wheel position — this is the core research insight.
- **Motor model:** `tau = Kt*(V - Kv*ω)/Rm` where Kv is configurable via `PhysicalParams(kv_override=...)`.
- **Physical params** come from MATLAB system ID in `Matlab/modelo_pendulo.m`. These MUST be used (not made up).

### PPO Configuration

- Network: policy [32, 32], value [64, 64] with **Tanh** activation (chosen for ESP32 fixed-point conversion)
- `ent_coef=0.005`: low entropy — let policy converge to position-dependent pattern
- `control_weight=0.005`: low because back-EMF naturally limits wheel speed
- `residual_scale=2.0V`: enough authority to compensate 0.05 Nm cogging

## Key Implementation Details

- **State vector:** `[theta, alpha, theta_dot, alpha_dot]` — theta=0 is upright, wrapped to [-π, π]
- **Firmware LQR gains** `[-5.5413, 0, -0.7263, -0.098]` differ from simulation gains `[-45.0, 0.0, -5.2, -0.62]` — firmware gains are too weak for the simulation plant
- **Firmware encoder conversion:** wheel `en2rad_roda = 0.006411413578755`, pendulum `en2rad_pend = 0.001570796326795`
- **Voltage saturation:** ±12V in both simulation and firmware
- **Communication protocol:** Binary frames with 0xFF header, 5 doubles, 0x00 footer
- **Termination:** |theta| > π/3 or |theta_dot| > 15 rad/s

## Results (1M steps PPO, IC: θ=0.05 rad)

- No-cogging LQR: **0.25°** RMS (target)
- Cogging LQR (A=0.05, N=7): **1.28°** RMS (5× degradation)
- Hybrid (LQR+RL, 2V): **0.64°** RMS (50.1% improvement, closes 62% of gap)
- RL uses ~0.8V steady-state (active compensation throughout, not just transient)
