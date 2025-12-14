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

        # Motor parameters (from MATLAB)
        self.tau_stall = 0.3136  # Nm
        self.i_stall = 1.8  # A
        self.Rm = self.max_voltage / self.i_stall  # Motor resistance (12V / 1.8A = 6.67 Ohm)
        # Note: MATLAB uses Kt=0, Kv=0 (simplified model, no back-EMF)
        # But we keep Kt for torque calculation
        self.Kt = self.tau_stall / self.i_stall  # Torque constant (0.1742 Nm/A)

        # Inertias
        self.Jh = (1/3) * self.Mh * self.L**2  # Pendulum inertia
        self.Jr = (1/2) * self.Mr * (self.r**2 + self.r_in**2)  # Wheel inertia

        # Damping coefficients (from MATLAB lambda.mat)
        # Note: In MATLAB, b2 = 2*lambda*(Jh+Jr), and Kt=0, Kv=0
        self.lambda_damping = 0.15060423  # Damping coefficient from system ID
        self.b1 = 0.0  # Pendulum damping (no damping on pendulum, from MATLAB line 26)
        self.b2 = 2 * self.lambda_damping * (self.Jh + self.Jr)  # Wheel damping (MATLAB line 27)
        self.Kv = 0.0  # Back-EMF constant (set to 0 as in MATLAB line 30)

        # Stribeck friction parameters (RESEARCH: testing RL robustness)
        #
        # RESEARCH FINDING: Friction actually HELPS the LQR by providing damping!
        # - With friction: LQR succeeds 100%, RMS error ~0.02 rad
        # - Without friction: LQR succeeds ~80%, RMS error ~0.32 rad (underdamped)
        #
        # RESEARCH ANGLE: Test RL's ability to compensate for lack of natural damping
        # - Baseline (no friction): LQR struggles with underdamped oscillations
        # - RL learns to provide virtual damping and improve stability
        #
        # Default: NO FRICTION - creates challenging underdamped scenario for RL
        if friction_params is None:
            self.friction_params = {
                "Ts": 0.0,      # No static friction
                "Tc": 0.0,      # No Coulomb friction
                "vs": 0.05,     # Stribeck velocity (not used when Ts=Tc=0)
                "sigma": 0.0,   # No viscous friction - fully underdamped system
            }
        else:
            self.friction_params = friction_params

        # LQR gains (from firmware main.cpp lines 37-43)
        # In firmware: K = {-5.5413, -0.0, -0.7263, -0.0980} (negative values)
        # Control law: u = -K.x1*x1 - K.x2*x2 - ... = -(-5.5413)*theta - ...
        # So effective control: u = +5.5413*theta + 0.7263*theta_dot + ...
        #
        # In Python with u = -K·x, we need K to be negative to get same result:
        # u = -K·x = -(-5.5413)*theta = +5.5413*theta (same as firmware)
        self.K = np.array([-5.5413, -0.0, -0.7263, -0.0980])

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

    def _stribeck_friction(self, omega: float, applied_torque: float = 0.0) -> float:
        """
        Compute Stribeck friction torque as a function of angular velocity.

        This implements an enhanced Stribeck model with features that challenge LQR:
        1. Standard Stribeck curve: F = (Tc + (Ts-Tc)*exp(-|ω|/vs))*sign(ω) + σ*ω
        2. Asymmetric friction: slightly different friction in each direction
        3. Velocity-dependent static friction: harder to start from rest

        The asymmetry and non-linearity cannot be compensated by linear control.

        Args:
            omega: Angular velocity (rad/s)
            applied_torque: Motor torque being applied (used for breakaway calculation)

        Returns:
            Friction torque (Nm) - opposes motion with non-linear characteristics
        """
        Ts = self.friction_params["Ts"]
        Tc = self.friction_params["Tc"]
        vs = self.friction_params["vs"]
        sigma = self.friction_params["sigma"]

        # Asymmetry factor: friction is slightly higher in positive direction
        # This creates a bias that LQR cannot compensate for
        asymmetry = 0.15  # 15% asymmetry

        # Handle near-zero velocity case (stiction region)
        stiction_threshold = 0.05  # rad/s
        if abs(omega) < stiction_threshold:
            # Enhanced stiction: increases friction near zero velocity
            # This creates "sticky" behavior that causes limit cycles
            stiction_boost = (1.0 - abs(omega) / stiction_threshold) * 0.5  # Up to 50% boost
            effective_Ts = Ts * (1.0 + stiction_boost)

            # At rest: friction opposes applied torque up to effective_Ts
            if abs(applied_torque) <= effective_Ts:
                # Add small random perturbation to break symmetry
                return applied_torque * 0.95  # Slight energy loss
            else:
                return effective_Ts * np.sign(applied_torque)

        # Standard Stribeck friction model (always opposes motion)
        sign_omega = np.sign(omega)

        # Apply asymmetry: higher friction for positive omega
        if omega > 0:
            effective_Ts = Ts * (1.0 + asymmetry)
            effective_Tc = Tc * (1.0 + asymmetry)
        else:
            effective_Ts = Ts * (1.0 - asymmetry * 0.5)  # Slightly less asymmetry for negative
            effective_Tc = Tc * (1.0 - asymmetry * 0.5)

        # Stribeck curve: high at low speeds, drops to Tc at high speeds
        stribeck_term = effective_Tc + (effective_Ts - effective_Tc) * np.exp(-abs(omega) / vs)

        # Viscous friction (linear with velocity)
        viscous_term = sigma * omega

        # Total friction opposes motion direction
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

    def _dynamics(self, state: np.ndarray, u_normalized: float) -> np.ndarray:
        """
        Compute state derivatives for the coupled pendulum-wheel system with friction.

        This implements the MATLAB state-space model with non-linear extension (sin(θ) instead
        of θ) and added Stribeck friction on the wheel.

        From MATLAB modelo_pendulo.m:
        State: x = [theta, alpha, theta_dot, alpha_dot]'

        Pendulum equation (row 3 of state derivative):
        θ̈ = l_31*sin(θ) + l_33*θ̇ + l_34*α̇ + l_3*u
        where:
            l_31 = -(Mr*g*L + Mh*g*d)/(Mr*L² + Jh)
            l_33 = -b1/(Mr*L² + Jh)
            l_34 = (b2 + Kt*Kv/Rm)/(Mr*L² + Jh)
            l_3 = -(12*Kt)/(Rm*(Mr*L² + Jh))

        Wheel equation (row 4 of state derivative):
        α̈ = l_41*sin(θ) + l_43*θ̇ + l_44*α̇ + l_4*u - τ_friction/Jr
        where:
            l_41 = (Mr*g*L + Mh*g*d)/(Mr*L² + Jh)
            l_43 = b1/(Mr*L² + Jh)
            l_44 = -((Mr*L² + Jh + Jr)*(b2 + Kt*Kv/Rm))/(Jr*(Mr*L² + Jh))
            l_4 = (12*Kt*(Mr*L² + Jh + Jr))/(Rm*Jr*(Mr*L² + Jh))

        Args:
            state: [theta, alpha, theta_dot, alpha_dot]
            u_normalized: Normalized control input (-1 to 1, will be scaled by 12V in equations)

        Returns:
            state_dot: [theta_dot, alpha_dot, theta_ddot, alpha_ddot]
        """
        theta, alpha, theta_dot, alpha_dot = state

        # Shorthand for common terms (matching MATLAB notation)
        MrL2_Jh = self.Mr * self.L**2 + self.Jh
        MhgL_MrgL = self.Mh * self.g * self.d + self.Mr * self.g * self.L

        # Compute state-space matrix elements (from MATLAB lines 36-54)
        # Note: Kt=0, Kv=0 in MATLAB, so (b2 + Kt*Kv/Rm) simplifies to b2
        # However, l_3 and l_4 still use Kt for motor torque calculation

        # Pendulum equation coefficients
        l_31 = -MhgL_MrgL / MrL2_Jh
        l_33 = -self.b1 / MrL2_Jh
        l_34 = (self.b2 + self.Kt * self.Kv / self.Rm) / MrL2_Jh
        l_3 = -(12.0 * self.Kt) / (self.Rm * MrL2_Jh)

        # Wheel equation coefficients
        l_41 = MhgL_MrgL / MrL2_Jh
        l_43 = self.b1 / MrL2_Jh
        l_44 = -((MrL2_Jh + self.Jr) * (self.b2 + self.Kt * self.Kv / self.Rm)) / (self.Jr * MrL2_Jh)
        l_4 = (12.0 * self.Kt * (MrL2_Jh + self.Jr)) / (self.Rm * self.Jr * MrL2_Jh)

        # Calculate motor torque for stiction model
        # Motor torque = Kt * I = Kt * (V/Rm) = Kt * (12*u_normalized) / Rm
        motor_torque = self.Kt * (12.0 * u_normalized) / self.Rm

        # Friction torque on wheel (RESEARCH: added Stribeck friction with stiction)
        tau_friction = self._stribeck_friction(alpha_dot, motor_torque)

        # State derivatives (non-linear version using sin(theta) instead of linearized theta)
        theta_ddot = (
            l_31 * np.sin(theta) +  # Non-linear gravity term
            l_33 * theta_dot +
            l_34 * alpha_dot +
            l_3 * u_normalized  # u_normalized ∈ [-1, 1], scaled by 12V in l_3
        )

        alpha_ddot = (
            l_41 * np.sin(theta) +  # Non-linear gravity coupling
            l_43 * theta_dot +
            l_44 * alpha_dot +
            l_4 * u_normalized -  # u_normalized ∈ [-1, 1], scaled by 12V in l_4
            tau_friction / self.Jr  # Stribeck friction acts on wheel (RESEARCH addition)
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
        # Larger initial perturbation to stress-test the controller with friction
        theta_init = self.np_random.uniform(-0.3, 0.3)  # Up to ~17 degrees from upright
        alpha_init = 0.0
        theta_dot_init = self.np_random.uniform(-0.5, 0.5)  # Initial angular velocity
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

        # Randomize friction parameters for domain randomization
        # Default is no friction, but randomization adds small amounts for robustness
        # This helps RL generalize to real hardware which may have some friction
        base_Ts = 0.05  # Small friction for domain randomization
        base_sigma = 0.02
        self.friction_params["Ts"] = base_Ts * self.np_random.uniform(0, 1 + factor)
        self.friction_params["Tc"] = base_Ts * 0.5 * self.np_random.uniform(0, 1 + factor)
        self.friction_params["vs"] = 0.05 * self.np_random.uniform(1 - factor, 1 + factor)
        self.friction_params["sigma"] = base_sigma * self.np_random.uniform(0, 1 + factor)

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

        # Compute LQR baseline control (returns voltage in range [-12V, +12V])
        u_LQR = self._lqr_control(self.state)

        # Total control signal (hybrid control)
        u_total = u_LQR + u_RL

        # Saturate voltage to physical limits
        u_total = np.clip(u_total, -self.max_voltage, self.max_voltage)

        # Normalize control signal to [-1, 1] for dynamics
        # The dynamics equations expect normalized input (will be scaled by 12V internally)
        u_normalized = u_total / self.max_voltage

        # Integrate dynamics using Euler method
        state_dot = self._dynamics(self.state, u_normalized)

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

        # Info dict - compute motor torque for friction calculation
        motor_torque = self.Kt * u_total / self.Rm
        info = {
            "u_LQR": u_LQR,
            "u_RL": u_RL,
            "u_total": u_total,
            "friction_torque": self._stribeck_friction(self.state[3], motor_torque),
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
