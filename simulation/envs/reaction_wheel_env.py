"""
Reaction Wheel Inverted Pendulum Gymnasium Environment

This environment simulates a reaction wheel inverted pendulum with Stribeck friction,
designed for training residual RL agents to compensate for non-linear friction effects.

The control architecture is hybrid:
    u_total = u_LQR + alpha * u_RL
where u_LQR is computed internally and u_RL is the agent's action.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class ReactionWheelEnv(gym.Env):
    """
    Reaction Wheel Inverted Pendulum Environment with Stribeck Friction

    State: [theta, alpha, theta_dot, alpha_dot]
        - theta: Pendulum angle (rad, 0 = upright, range: [-pi, pi])
        - alpha: Wheel angle (rad)
        - theta_dot: Pendulum angular velocity (rad/s)
        - alpha_dot: Wheel angular velocity (rad/s)

    Action: Residual control signal (scalar)
        - The agent outputs a residual u_RL that is added to the base LQR controller

    Physical Parameters (from MATLAB system identification):
        - g = 9.81 m/s²
        - Mh = 0.149 kg (pendulum mass)
        - Mr = 0.144 kg (wheel mass)
        - L = 0.14298 m (pendulum COM to pivot)
        - d = 0.0987 m
        - Motor stall torque = 0.3136 Nm @ 1.8A
        - Max voltage = 12V
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        dt: float = 0.02,  # 50 Hz control frequency
        max_voltage: float = 12.0,
        residual_scale: float = 1.0,  # Scaling factor alpha for residual action
        friction_params: Optional[Dict[str, float]] = None,
        domain_randomization: bool = False,
        randomization_factor: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.dt = dt
        self.max_voltage = max_voltage
        self.residual_scale = residual_scale
        self.domain_randomization = domain_randomization
        self.randomization_factor = randomization_factor
        self.render_mode = render_mode

        # Physical parameters (from MATLAB system ID)
        self.g = 9.81  # m/s²
        self.Mh = 0.149  # kg - pendulum mass
        self.Mr = 0.144  # kg - wheel mass
        self.L = 0.14298  # m - pendulum length (COM to pivot)
        self.d = 0.0987  # m

        # Wheel geometry
        self.r = 0.1  # m - outer radius
        self.r_in = 0.0911  # m - inner radius

        # Motor parameters
        self.tau_stall = 0.3136  # Nm
        self.i_stall = 1.8  # A
        self.Rm = self.max_voltage / self.i_stall  # Motor resistance
        self.Kt = self.tau_stall / self.i_stall  # Torque constant

        # Inertias
        self.Jh = (1/3) * self.Mh * self.L**2  # Pendulum inertia
        self.Jr = (1/2) * self.Mr * (self.r**2 + self.r_in**2)  # Wheel inertia

        # Damping (assumed minimal for now, can be tuned)
        self.b1 = 0.0  # Pendulum damping
        self.b2 = 0.0  # Wheel damping

        # Stribeck friction parameters (to be tuned based on real hardware)
        if friction_params is None:
            self.friction_params = {
                "Ts": 0.15,      # Static friction (Nm)
                "Tc": 0.08,      # Coulomb friction (Nm)
                "vs": 0.05,      # Stribeck velocity (rad/s)
                "sigma": 0.02,   # Viscous friction coefficient
            }
        else:
            self.friction_params = friction_params

        # LQR gains (from firmware: K = [-5.5413, -0.0, -0.7263, -0.0980])
        self.K = np.array([5.5413, 0.0, 0.7263, 0.0980])

        # State limits for observation space
        # theta: [-pi, pi], alpha: unbounded but normalized, velocities: reasonable limits
        max_theta_dot = 10.0  # rad/s
        max_alpha_dot = 50.0  # rad/s (wheel spins faster)

        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -max_theta_dot, -max_alpha_dot]),
            high=np.array([np.pi, np.inf, max_theta_dot, max_alpha_dot]),
            dtype=np.float32,
        )

        # Action space: residual control signal (will be scaled and added to LQR)
        # Normalized to [-1, 1], then scaled by residual_scale
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # State variables
        self.state = None
        self.steps = 0
        self.max_steps = 500  # 10 seconds at 50Hz

        # For rendering
        self.screen = None
        self.clock = None

    def _stribeck_friction(self, omega: float) -> float:
        """
        Compute Stribeck friction torque as a function of angular velocity.

        F_friction = (Tc + (Ts - Tc) * exp(-|ω|/vs)) * sign(ω) + σ*ω

        Args:
            omega: Angular velocity (rad/s)

        Returns:
            Friction torque (Nm)
        """
        Ts = self.friction_params["Ts"]
        Tc = self.friction_params["Tc"]
        vs = self.friction_params["vs"]
        sigma = self.friction_params["sigma"]

        # Stribeck model
        sign_omega = np.sign(omega) if omega != 0 else 0
        stribeck_term = Tc + (Ts - Tc) * np.exp(-np.abs(omega) / vs)
        viscous_term = sigma * omega

        return stribeck_term * sign_omega + viscous_term

    def _lqr_control(self, state: np.ndarray) -> float:
        """
        Compute LQR control signal.

        u_LQR = -K · x

        Args:
            state: [theta, alpha, theta_dot, alpha_dot]

        Returns:
            Control voltage (V)
        """
        return -np.dot(self.K, state)

    def _dynamics(self, state: np.ndarray, u_total: float) -> np.ndarray:
        """
        Compute state derivatives for the coupled pendulum-wheel system with friction.

        Equations of motion derived from Lagrangian mechanics:
        (Mh*d + Mr*L)*L*θ̈ + Jr*α̈ = (Mh*d + Mr*L)*g*sin(θ) + τ_friction - τ_motor
        Jr*α̈ + (Mh*d + Mr*L)*L*θ̈*cos(θ) = τ_motor - τ_friction

        Args:
            state: [theta, alpha, theta_dot, alpha_dot]
            u_total: Total control voltage (V)

        Returns:
            state_dot: [theta_dot, alpha_dot, theta_ddot, alpha_ddot]
        """
        theta, alpha, theta_dot, alpha_dot = state

        # Motor torque (simplified model: τ = Kt * i = Kt * u / Rm)
        tau_motor = (self.Kt / self.Rm) * u_total * 12.0  # Scale by max voltage

        # Friction torque on wheel
        tau_friction = self._stribeck_friction(alpha_dot)

        # Shorthand for common terms
        MrL2_Jh = self.Mr * self.L**2 + self.Jh
        MhgL_MrgL = self.Mh * self.g * self.d + self.Mr * self.g * self.L

        # Mass matrix and force vector for the coupled system
        # Using the linearized version from MATLAB as a starting point,
        # but this is the full non-linear version

        # Simplified non-linear equations (assuming small angle approximations can be relaxed)
        # From the state-space in MATLAB, we can derive:

        # Pendulum equation:
        # (Mr*L² + Jh)*θ̈ = (Mh*g*d + Mr*g*L)*sin(θ) - b1*θ̇ + (b2 + Kt*Kv/Rm)*α̇ - (Kt/Rm)*u*12

        # Wheel equation:
        # Jr*α̈ = -(Mh*g*d + Mr*g*L)*sin(θ) + b1*θ̇ - ((Jr + Mr*L² + Jh)/Jr)*(b2 + Kt*Kv/Rm)*α̇ + ((Jr + Mr*L² + Jh)/Jr)*(Kt/Rm)*u*12

        # For the coupled system with friction:
        theta_ddot = (
            MhgL_MrgL * np.sin(theta) / MrL2_Jh
            - self.b1 * theta_dot / MrL2_Jh
            + (self.b2 * self.Kt / self.Rm) * alpha_dot / MrL2_Jh
            - (self.Kt / self.Rm) * u_total * 12.0 / MrL2_Jh
        )

        alpha_ddot = (
            -MhgL_MrgL * np.sin(theta) / self.Jr
            + self.b1 * theta_dot / self.Jr
            - ((self.Jr + MrL2_Jh) / self.Jr) * (self.b2 * self.Kt / self.Rm) * alpha_dot
            + ((self.Jr + MrL2_Jh) / self.Jr) * (self.Kt / self.Rm) * u_total * 12.0
            - tau_friction / self.Jr  # Friction acts on wheel
        )

        return np.array([theta_dot, alpha_dot, theta_ddot, alpha_ddot])

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-pi, pi] range.

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle in [-pi, pi]
        """
        # Wrap to [0, 2*pi]
        angle = angle % (2 * np.pi)
        # Shift to [-pi, pi]
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Initial state: pendulum near upright, wheel at rest
        # Add small random perturbation for exploration
        theta_init = self.np_random.uniform(-0.1, 0.1)  # Near upright
        alpha_init = 0.0
        theta_dot_init = self.np_random.uniform(-0.1, 0.1)
        alpha_dot_init = 0.0

        self.state = np.array([theta_init, alpha_init, theta_dot_init, alpha_dot_init])
        self.steps = 0

        # Apply domain randomization if enabled
        if self.domain_randomization:
            self._randomize_parameters()

        return self.state.astype(np.float32), {}

    def _randomize_parameters(self):
        """Apply domain randomization to physical parameters for robust sim-to-real transfer."""
        factor = self.randomization_factor

        # Randomize masses
        self.Mh = 0.149 * self.np_random.uniform(1 - factor, 1 + factor)
        self.Mr = 0.144 * self.np_random.uniform(1 - factor, 1 + factor)

        # Randomize lengths
        self.L = 0.14298 * self.np_random.uniform(1 - factor, 1 + factor)
        self.d = 0.0987 * self.np_random.uniform(1 - factor, 1 + factor)

        # Recalculate inertias
        self.Jh = (1/3) * self.Mh * self.L**2
        self.Jr = (1/2) * self.Mr * (self.r**2 + self.r_in**2)

        # Randomize friction parameters
        self.friction_params["Ts"] = 0.15 * self.np_random.uniform(1 - factor, 1 + factor)
        self.friction_params["Tc"] = 0.08 * self.np_random.uniform(1 - factor, 1 + factor)
        self.friction_params["vs"] = 0.05 * self.np_random.uniform(1 - factor, 1 + factor)
        self.friction_params["sigma"] = 0.02 * self.np_random.uniform(1 - factor, 1 + factor)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment.

        Args:
            action: Residual control signal from RL agent (normalized to [-1, 1])

        Returns:
            observation: Next state
            reward: Reward for this transition
            terminated: Whether episode ended due to failure
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # Extract residual action and scale it
        u_RL = float(action[0]) * self.residual_scale

        # Compute LQR baseline control
        u_LQR = self._lqr_control(self.state)

        # Total control signal (hybrid control)
        u_total = u_LQR + u_RL

        # Saturate voltage to physical limits
        u_total = np.clip(u_total, -self.max_voltage, self.max_voltage)

        # Integrate dynamics using RK4 for better accuracy
        state_dot = self._dynamics(self.state, u_total)

        # Simple Euler integration (can be upgraded to RK4 if needed)
        self.state = self.state + state_dot * self.dt

        # Normalize pendulum angle to [-pi, pi]
        self.state[0] = self._normalize_angle(self.state[0])

        # Compute reward (MRAC-based: penalize deviation from ideal behavior)
        reward = self._compute_reward(self.state, u_LQR, u_RL)

        # Check termination conditions
        terminated = self._is_terminated(self.state)

        self.steps += 1
        truncated = self.steps >= self.max_steps

        # Info dict
        info = {
            "u_LQR": u_LQR,
            "u_RL": u_RL,
            "u_total": u_total,
            "friction_torque": self._stribeck_friction(self.state[3]),
        }

        return self.state.astype(np.float32), reward, terminated, truncated, info

    def _compute_reward(self, state: np.ndarray, u_LQR: float, u_RL: float) -> float:
        """
        MRAC-based reward: penalize deviation from ideal reference behavior.

        The ideal reference model is the frictionless LQR-controlled system.
        We penalize:
        1. State deviation from upright equilibrium
        2. Control effort (especially residual)
        3. Large velocities

        Args:
            state: Current state [theta, alpha, theta_dot, alpha_dot]
            u_LQR: LQR control signal
            u_RL: Residual RL control signal

        Returns:
            reward: Scalar reward
        """
        theta, alpha, theta_dot, alpha_dot = state

        # Penalize angle deviation from upright (theta = 0)
        angle_cost = theta**2

        # Penalize angular velocities
        velocity_cost = 0.1 * (theta_dot**2 + 0.01 * alpha_dot**2)

        # Penalize residual control effort (encourage minimal intervention)
        control_cost = 0.01 * u_RL**2

        # Total reward (negative cost)
        reward = -(angle_cost + velocity_cost + control_cost)

        # Bonus for staying upright
        if abs(theta) < 0.1:
            reward += 1.0

        return reward

    def _is_terminated(self, state: np.ndarray) -> bool:
        """
        Check if episode should terminate due to failure.

        Args:
            state: Current state

        Returns:
            True if episode should end
        """
        theta, _, theta_dot, _ = state

        # Terminate if pendulum falls too far
        if abs(theta) > np.pi / 3:  # More than 60 degrees from upright
            return True

        # Terminate if spinning too fast (unstable)
        if abs(theta_dot) > 15.0:
            return True

        return False

    def render(self):
        """Render the environment (placeholder for future visualization)."""
        if self.render_mode == "human":
            # TODO: Implement visualization using pygame or matplotlib
            pass

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
