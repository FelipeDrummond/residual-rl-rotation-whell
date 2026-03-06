# Training Guide: Residual RL for Reaction Wheel Pendulum

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python -m simulation.train --timesteps 500000 --save_path models/ppo_virtual_damping
```

### 3. Validate Results
```bash
python -m simulation.validate --model_path models/ppo_virtual_damping/final_model
```

---

## Research Context: Virtual Damping

The training environment uses an **underdamped LQR** configuration. The key insight is:

**Friction HELPS the LQR by providing natural damping!**

Without friction, the system is underdamped and oscillates. The RL agent learns to provide **virtual damping** (`u_RL ∝ -ω`) that replaces unreliable physical friction.

### Why This is Compelling

1. **Genuine problem**: Underdamped oscillations are common in control systems
2. **LQR still essential**: Handles the core stabilization task
3. **RL role is clear**: Learns a damping function (simple, interpretable)
4. **Practical benefit**: Virtual damping is more reliable than physical friction

---

## Environment Configuration

The training environment uses the "underdamped LQR" configuration:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lqr_gain_scale` | 0.35 | Moderate gains → underdamped response |
| `observation_noise_std` | 0.0 | No artificial noise |
| `disturbance_std` | 0.0 | No artificial disturbances |
| Friction | None | No physical friction (isolate damping problem) |
| `residual_scale` | 2.0 | RL authority for virtual damping |

### Baseline Performance

With this configuration:
- **Underdamped LQR alone:** 100% survival, ~3° RMS error (oscillatory)
- **Target (Hybrid):** 100% survival, <1° RMS error (damped)
- **Reference (Optimal LQR, scale=1.0):** 100% survival, ~0.9° RMS error

---

## Training Command Options

```bash
python -m simulation.train [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--timesteps` | 500,000 | Total training timesteps |
| `--n_envs` | 4 | Number of parallel environments |
| `--learning_rate` | 3e-4 | PPO learning rate |
| `--residual_scale` | 2.0 | Scaling factor for RL action (virtual damping authority) |
| `--no_domain_rand` | False | Disable domain randomization |
| `--save_path` | models/ppo_residual | Model save directory |
| `--tensorboard_log` | ./logs/ | TensorBoard log directory |
| `--device` | auto | Device (auto/cuda/mps/cpu) |
| `--cpu` | False | Force CPU usage |

### Example Configurations

**Fast training (testing):**
```bash
python -m simulation.train --timesteps 100000 --n_envs 8
```

**Production training:**
```bash
python -m simulation.train --timesteps 1000000 --n_envs 8 --save_path models/production
```

**CPU-only (no GPU):**
```bash
python -m simulation.train --cpu --n_envs 4
```

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./logs/
```

Then open http://localhost:6006 in your browser.

**Key metrics to watch:**
- `rollout/ep_rew_mean` - Episode reward (should increase)
- `rollout/ep_len_mean` - Episode length (should approach 500)
- `train/value_loss` - Value function loss (should decrease)
- `train/policy_gradient_loss` - Policy loss (should stabilize)

### Learning Curves

Training automatically generates learning curve plots in the save directory:
- `learning_curves.png` - Episode rewards and lengths over time

---

## PPO Hyperparameters

The training script uses these PPO settings:

```python
PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,          # Steps per rollout
    batch_size=64,
    n_epochs=10,           # PPO update epochs
    gamma=0.995,           # Discount factor (high for stability tasks)
    gae_lambda=0.95,       # GAE parameter
    clip_range=0.2,        # PPO clip range
    ent_coef=0.005,        # Entropy coefficient (low for consistent behavior)
    vf_coef=0.5,           # Value function coefficient
    max_grad_norm=0.5,     # Gradient clipping
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),  # Small network for ESP32
        activation_fn=nn.Tanh,  # Tanh for fixed-point conversion
    ),
)
```

### Why These Settings?

- **High gamma (0.995):** Stability is a long-horizon task
- **Low entropy (0.005):** Damping requires consistent behavior, not exploration
- **Small network (64x64):** Must fit on ESP32 for deployment
- **Tanh activation:** Easier to convert to fixed-point arithmetic

---

## Validation

### Command Options

```bash
python -m simulation.validate [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model_path` | models/ppo_residual/final_model | Path to trained model |
| `--episodes` | 50 | Number of evaluation episodes |
| `--max_steps` | 500 | Maximum steps per episode |
| `--output_dir` | validation_results | Output directory for plots |
| `--no_show` | False | Don't display plots (save only) |

### Output Files

Validation generates:
- `comparison_plots.png` - Time series comparison
- `phase_portrait.png` - Stability analysis (θ vs θ̇)

### Interpreting Results

**Good results look like:**
- Episode length: 500 (full episode survival)
- RMS error: <1° for hybrid (vs ~3° for underdamped LQR)
- Control effort: Moderate (not constant saturation)
- Phase portrait: Tight spiral to origin (hybrid) vs oscillatory (LQR)

**Bad results look like:**
- Short episode length (early termination)
- RMS error higher than LQR baseline
- Constant saturation (bang-bang control)
- Phase portrait: Chaotic or limit cycles

### What the RL Should Learn

The RL agent should learn approximately:
```
u_RL(state) ≈ -k * alpha_dot
```

This is equivalent to viscous friction (velocity-proportional damping). You can verify this by plotting `u_RL` vs `alpha_dot` - it should show a negative linear relationship.

---

## Domain Randomization

When enabled (default), training randomizes physical parameters for robust sim-to-real transfer:

| Parameter | Randomization Range |
|-----------|-------------------|
| Masses (Mh, Mr) | ±10% |
| Lengths (L, d) | ±10% |
| LQR gain scale | ±10% around 0.35 |

**Note:** Friction is NOT randomized in the virtual damping scenario since we train without friction.

---

## Checkpoints

Training saves checkpoints at regular intervals:
- `models/ppo_residual/ppo_checkpoint_50000_steps.zip`
- `models/ppo_residual/ppo_checkpoint_100000_steps.zip`
- ...
- `models/ppo_residual/final_model.zip`
- `models/ppo_residual/vec_normalize.pkl` (observation normalization)

### Loading a Checkpoint

```python
from stable_baselines3 import PPO

model = PPO.load("models/ppo_residual/ppo_checkpoint_100000_steps")
```

### Resuming Training

Currently not implemented. Training always starts fresh.

---

## Troubleshooting

### Episode Length Stuck at Low Value

**Symptom:** `ep_len_mean` stays around 30-50, never increases.

**Causes:**
1. Initial conditions too extreme
2. LQR gain scale too low
3. Learning rate too high

**Solutions:**
1. Check initial condition ranges in config
2. Verify `lqr_gain_scale >= 0.25` (minimum for survival)
3. Try `--learning_rate 1e-4`

### No Improvement Over Baseline

**Symptom:** Hybrid performs same as or worse than underdamped LQR.

**Causes:**
1. Training not converged
2. Residual scale too low
3. LQR gains too high (already well-damped)

**Solutions:**
1. Train longer (1M+ steps)
2. Increase `--residual_scale` to 3.0-4.0
3. Verify `lqr_gain_scale=0.35` (not 1.0)

### Oscillatory Behavior Not Damped

**Symptom:** Hybrid still oscillates like underdamped LQR.

**Causes:**
1. RL not learning the damping function
2. Residual scale too low for effective damping

**Solutions:**
1. Check if `u_RL` correlates with `-alpha_dot`
2. Increase `--residual_scale`
3. Train longer

### Out of Memory

**Symptom:** CUDA/MPS out of memory error.

**Solutions:**
1. Reduce `--n_envs`
2. Use `--cpu` flag
3. Reduce `n_steps` in PPO config

---

## Hardware Acceleration

The training script auto-detects available hardware:

| Hardware | Detection | Performance |
|----------|-----------|-------------|
| NVIDIA GPU | `torch.cuda.is_available()` | Fastest |
| Apple Silicon | `torch.backends.mps.is_available()` | Fast |
| CPU | Fallback | Slowest |

**Expected training times (500K steps, n_envs=4):**
- NVIDIA GPU (RTX 3080): ~20 minutes
- Apple M1/M2: ~30 minutes
- CPU (8 cores): ~60 minutes

---

## Next Steps After Training

1. **Validate** - Run validation script to check performance
2. **Verify Damping** - Plot `u_RL` vs `alpha_dot` to confirm damping learned
3. **Analyze** - Look at phase portraits and control signals
4. **Export** - Convert model to ONNX or C arrays for ESP32
5. **Deploy** - Transfer to hardware (see firmware documentation)
