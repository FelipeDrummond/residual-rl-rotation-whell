# Setup Guide

## Python Environment Setup

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
```

2. **Activate the virtual environment:**
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note for GPU Acceleration:**
- **Apple Silicon (M1/M2/M3)**: PyTorch with MPS support is included automatically (PyTorch 2.0+)
- **NVIDIA GPU**: If you have a CUDA-capable GPU, consider installing PyTorch with CUDA support first:
  ```bash
  # Visit https://pytorch.org/get-started/locally/ for the latest CUDA version
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  ```
- **CPU only**: The default installation works fine on CPU

4. **Verify installation:**
```bash
python -m simulation.test_env
```

5. **Check GPU availability** (optional):
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('MPS available:', torch.backends.mps.is_available())"
```

If all tests pass, you're ready to start training!

## Quick Start

### Test the Environment
```bash
python -m simulation.test_env
```

### Train an Agent
```bash
python -m simulation.train --timesteps 500000
```

### Validate Results
```bash
python -m simulation.validate --model_path models/ppo_residual/final_model
```

### Monitor Training (in separate terminal)
```bash
tensorboard --logdir ./logs/
```

## Firmware Setup (ESP32)

See [CLAUDE.md](CLAUDE.md) for firmware build instructions using PlatformIO.

## Troubleshooting

**ImportError: No module named 'gymnasium'**
- Make sure you've activated the virtual environment and installed requirements

**Module not found: 'simulation'**
- Make sure you're running commands from the repository root directory

**CUDA/GPU errors (optional)**
- The code works fine on CPU. If you want GPU acceleration, install pytorch with CUDA support first.
