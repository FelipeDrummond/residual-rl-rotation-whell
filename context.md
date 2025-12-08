# Project Context: Residual RL for Reaction Wheel Pendulum

## 1. Project Objective
I am working on a research paper titled **"Hybrid Control for Reaction Wheel Pendulums: Non-Linear Friction Compensation via Residual Reinforcement Learning."**

The goal is to control a physical Reaction Wheel Inverted Pendulum using an ESP32.
* **The Problem:** The physical plant has significant non-linear Stribeck friction (stiction) that causes the standard LQR controller to oscillate (limit cycles) or fail to settle.
* **The Solution:** Implement a **Hybrid Controller**: $u_{total} = u_{LQR} + u_{RL}$.
    * The LQR handles the nominal linear dynamics.
    * The RL agent (PPO) learns to estimate and cancel the non-linear friction residue.
* **The Method:** We are using a **Sim-to-Real** workflow. We must first build a high-fidelity Python simulation (Digital Twin) that reproduces the "bad" friction behavior to train the agent before deploying to the ESP32.

## 2. Technical Architecture
* **Simulation:** Python + `gymnasium`.
* **RL Algorithm:** PPO (Proximal Policy Optimization) via `stable-baselines3`.
* **Reward Function Strategy:** Model Reference Adaptive Control (MRAC). The reward should penalize the difference between the *Real Plant* (with friction) and an *Ideal Reference Model* (frictionless LQR).
* **Target Hardware:** ESP32 (running C++). The RL policy must be small enough (MLP) to be exported to C arrays for inference.

## 3. Physical Parameters (From MATLAB System ID)
These are the verified parameters of the physical plant:
* **Gravity (g):** 9.81 m/s²
* **Mass Pendulum (Mh):** 0.149 kg
* **Mass Wheel (Mr):** 0.144 kg
* **Length (L):** 0.14298 m (COM to pivot)
* **Distance (d):** 0.0987 m
* **Motor Stall Torque:** 0.3136 Nm
* **Stall Current:** 1.8 A
* **Max Voltage:** 12.0 V
* **Derived Inertias:**
    * `Jh` (Pendulum) = 1/3 * Mh * L^2
    * `Jr` (Wheel) = 1/2 * Mr * (r^2 + r_in^2) where r=0.1m, r_in=0.0911m
* **Friction Model:** The simulation MUST include Stribeck Friction on the wheel:
    * `F_friction = (Tc + (Ts - Tc) * exp(-|w|/vs)) * sign(w) + sigma*w`
    * *Note:* `Ts` (Static) > `Tc` (Coulomb).

## 4. Current Control Logic (LQR Baseline)
We already have a tuned LQR controller working on the ESP32 (though it oscillates due to friction).
* **State Vector:** `[theta (rad), alpha (wheel rad), theta_dot, alpha_dot]`
* **LQR Gains (K):** `[-5.5413, -0.0, -0.7263, -0.0980]`
* **Control Frequency:** 50Hz (dt = 0.02s)

## 5. Immediate Task
I need you to help me build the **Python Gymnasium Environment** (`ReactionWheelEnv`) for this project.
The environment must:
1.  Implement the non-linear physics equations of motion for the coupled pendulum-wheel system.
2.  Include the **Stribeck Friction** function to simulate the hardware degradation.
3.  Include the **LQR controller** inside the step function (the agent only adds a residual $\Delta u$).
4.  Return the state `[theta, alpha, theta_dot, alpha_dot]`.