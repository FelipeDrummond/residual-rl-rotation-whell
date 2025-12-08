# Reaction Wheel Pendulum - Python Simulation

This directory contains the Python digital twin for training residual RL agents to compensate for Stribeck friction in the reaction wheel inverted pendulum.

## Setup

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Verify installation:
```bash
python -c "from simulation.envs import ReactionWheelEnv; print('Environment loaded successfully!')"
```

## Training

Train a PPO agent with default parameters (auto-detects GPU):
```bash
python -m simulation.train
```

The training script automatically detects the best available device:
- **Apple Silicon (M1/M2/M3)**: Uses MPS (Metal Performance Shaders)
- **NVIDIA GPU**: Uses CUDA
- **CPU**: Falls back to CPU

Training with custom parameters:
```bash
python -m simulation.train \
    --timesteps 1000000 \
    --n_envs 8 \
    --learning_rate 3e-4 \
    --residual_scale 0.5 \
    --save_path models/my_model
```

Force specific device:
```bash
# Force CPU (useful for debugging)
python -m simulation.train --cpu

# Or specify device explicitly
python -m simulation.train --device mps   # Apple Silicon
python -m simulation.train --device cuda  # NVIDIA GPU
python -m simulation.train --device cpu   # CPU only
```

Arguments:
- `--timesteps`: Total training timesteps (default: 500,000)
- `--n_envs`: Number of parallel environments (default: 4)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--residual_scale`: Scaling factor α for residual action (default: 1.0)
- `--no_domain_rand`: Disable domain randomization
- `--save_path`: Model save path (default: models/ppo_residual)
- `--device`: Device selection: auto/cuda/mps/cpu (default: auto)
- `--cpu`: Shorthand to force CPU usage

Monitor training with TensorBoard:
```bash
tensorboard --logdir ./logs/
```

## Validation

Compare LQR vs Hybrid (LQR + RL) control:
```bash
python -m simulation.validate \
    --model_path models/ppo_residual/final_model \
    --episodes 10
```

This will:
1. Evaluate pure LQR control (with friction)
2. Evaluate trained hybrid controller
3. Print performance metrics (RMS error, reward, episode length)
4. Generate comparison plots saved to `validation_plots.png`

## Environment Details

### ReactionWheelEnv

**Observation Space:** `[theta, alpha, theta_dot, alpha_dot]`
- `theta`: Pendulum angle (rad, 0 = upright)
- `alpha`: Wheel angle (rad)
- `theta_dot`: Pendulum angular velocity (rad/s)
- `alpha_dot`: Wheel angular velocity (rad/s)

**Action Space:** Residual control signal (scalar, normalized to [-1, 1])

**Reward Function:** MRAC-based reward that penalizes:
- Angle deviation from upright: `theta²`
- Angular velocities: `0.1 * (theta_dot² + 0.01 * alpha_dot²)`
- Control effort: `0.01 * u_RL²`
- Bonus: +1.0 for staying within ±0.1 rad of upright

**Physics:**
- Non-linear equations of motion for coupled pendulum-wheel system
- Stribeck friction model on wheel:
  ```
  F_friction = (Tc + (Ts - Tc) * exp(-|ω|/vs)) * sign(ω) + σ*ω
  ```
- Default friction parameters (tunable):
  - `Ts = 0.15 Nm` (static friction)
  - `Tc = 0.08 Nm` (Coulomb friction)
  - `vs = 0.05 rad/s` (Stribeck velocity)
  - `sigma = 0.02` (viscous friction)

**Control Architecture:**
```
u_total = u_LQR + α * u_RL
```
where:
- `u_LQR = -K · x` (computed internally)
- `u_RL` is the agent's action
- `α` is the residual scale factor

**Domain Randomization:** When enabled, randomizes:
- Masses (±10%)
- Lengths (±10%)
- Friction parameters (±10%)

## Next Steps

After training:
1. Export model to ONNX format for deployment
2. Convert network weights to C arrays for ESP32
3. Integrate with firmware in `firmware/src/main.cpp`

See the main [CLAUDE.md](../CLAUDE.md) for integration details.
