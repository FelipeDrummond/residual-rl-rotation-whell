# Residual RL for Reaction Wheel Friction Compensation

## 📝 Abstract
This project explores a **Hybrid Control Architecture** (Residual Reinforcement Learning) applied to a **Reaction Wheel Inverted Pendulum**.

Standard Linear Quadratic Regulators (LQR) often fail to stabilize low-cost robotic hardware due to significant non-linearities, specifically **Stribeck Friction** and **Dead Zones** in the actuator, leading to steady-state errors or limit cycle oscillations ("jitter").

This repository implements a **Residual RL** agent that learns to estimate and cancel these specific non-linearities, allowing the base LQR controller to operate on a "linearized" virtual plant. The system is trained in a custom `Gymnasium` environment with domain randomization and deployed to an ESP32 microcontroller.

## 🏗️ Architecture

The control law follows the residual formulation:

$$u_{total}(t) = u_{LQR}(t) + \alpha \cdot \pi_{\theta}(s_t)$$

- **$u_{LQR}$**: Base linear controller (stabilizes the nominal model).
- **$\pi_{\theta}$**: PPO Agent (compensates for model mismatch/friction).
- **$\alpha$**: Scaling factor for safety.



## 📂 Repository Structure

```bash
├── firmware/           # C++ code for ESP32 (LQR + Neural Net Inference)
│   ├── src/main.cpp
│   └── include/
├── simulation/         # Python Digital Twin & RL Training
│   ├── envs/           # Custom Gymnasium Environment (Stribeck Model)
│   ├── train.py        # Stable Baselines3 PPO Training Script
│   └── validate.py     # Comparison plots (LQR vs Hybrid)
├── matlab/             # Legacy system identification & LQR tuning
│   ├── modelo_pendulo.m
│   └── lambda.mat
└── models/             # Trained ONNX/Zip models