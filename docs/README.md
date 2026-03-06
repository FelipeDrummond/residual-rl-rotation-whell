# Documentation Index

This directory contains detailed documentation of the research findings and implementation details for the Residual RL Reaction Wheel Pendulum project.

## Documents

### [RESEARCH_NOTES.md](RESEARCH_NOTES.md)
**Comprehensive research journal** documenting the timeline of discoveries, including:
- Initial LQR oscillation bug and fix
- Extreme friction experiments (failed approach)
- Combined challenges approach (deprecated)
- **Virtual damping approach (current method)**

### [PHYSICS_MODEL.md](PHYSICS_MODEL.md)
**Detailed physics documentation** covering:
- State-space model derivation
- Physical parameters from MATLAB system ID
- Equations of motion
- Sign conventions (critical!)
- Numerical values

### [FRICTION_ANALYSIS.md](FRICTION_ANALYSIS.md)
**Analysis of friction experiments** including:
- Stribeck friction model
- Key finding: friction HELPS by providing damping
- Why extreme friction failed as a research scenario
- Transition to virtual damping approach

### [LQR_DEBUGGING.md](LQR_DEBUGGING.md)
**Debugging guide for LQR instability**:
- Root cause analysis (sign convention mismatch)
- MATLAB vs Python conventions
- Gain computation
- Debugging checklist

### [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
**Practical guide for training and validation**:
- Quick start commands
- Environment configuration
- PPO hyperparameters
- TensorBoard monitoring
- Troubleshooting common issues

---

## Quick Reference

### Key Finding: The Sign Convention Bug

The MATLAB model uses `theta=0` at the bottom (hanging), while our simulation uses `theta=0` at the top (inverted). This required inverting the gravity term sign:

```python
# Correct for inverted pendulum (theta=0 upright):
l_31 = +MhgL_MrgL / MrL2_Jh  # Positive = destabilizing
```

### Key Finding: Friction HELPS, Not Hurts!

Experiments revealed that friction provides beneficial damping:

| LQR Gain Scale | No Friction | With Friction |
|----------------|-------------|---------------|
| 0.25 | 0% success | 100% success |
| 0.30 | 100%, 12.7° RMS | 100%, 1.8° RMS |
| 0.35 | 100%, 3.0° RMS | 100%, 1.3° RMS |
| 1.00 | 100%, 0.9° RMS | 100%, 0.8° RMS |

**Insight:** The underdamped system (no friction) oscillates. Friction provides natural damping.

### Current Approach: Virtual Damping

The RL agent learns to provide **virtual damping** (`u_RL ∝ -ω`) for an underdamped LQR:

| Configuration | Description |
|---------------|-------------|
| `lqr_gain_scale=0.35` | Moderate gains → underdamped response |
| `friction=None` | No physical friction (isolate damping problem) |
| `residual_scale=2.0` | RL authority for damping |

**Expected Results:**
- Underdamped LQR alone: 100% survival, ~3° RMS (oscillatory)
- Hybrid (LQR + RL): 100% survival, <1° RMS (RL provides damping)

**Why This is Compelling:**
1. Genuine problem (underdamped oscillations are common)
2. LQR is essential (handles stabilization)
3. RL role is clear (learns damping function)
4. Practical benefit (more reliable than physical friction)

---

## Training Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (500K steps, auto GPU detection)
python -m simulation.train --timesteps 500000 --save_path models/ppo_virtual_damping

# Monitor training
tensorboard --logdir ./logs/

# Validate
python -m simulation.validate --model_path models/ppo_virtual_damping/final_model

# Tune friction parameters (for experiments)
python -m simulation.tune_friction --plot
```

---

## File Structure

```
docs/
├── README.md              # This index file
├── RESEARCH_NOTES.md      # Research journal
├── PHYSICS_MODEL.md       # Physics documentation
├── FRICTION_ANALYSIS.md   # Friction experiments
├── LQR_DEBUGGING.md       # LQR debugging guide
└── TRAINING_GUIDE.md      # Training instructions
```

---

## Related Files

- `CLAUDE.md` - Project overview and AI assistant instructions
- `simulation/config.py` - Centralized configuration
- `simulation/envs/reaction_wheel_env.py` - Environment implementation
- `simulation/train.py` - Training script
- `simulation/validate.py` - Validation script
- `simulation/tune_friction.py` - Friction parameter tuning
- `Matlab/modelo_pendulo.m` - MATLAB system ID model
