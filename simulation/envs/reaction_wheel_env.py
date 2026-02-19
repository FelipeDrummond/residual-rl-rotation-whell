"""
Reaction Wheel Inverted Pendulum Gymnasium Environment

This environment simulates a reaction wheel inverted pendulum with Stribeck friction,
designed for training residual RL agents to compensate for stiction dead zones.

Stribeck friction creates a dead zone where the motor cannot move the wheel at small
pendulum angles (LQR commands too weak to overcome stiction). The RL agent learns
supplemental torque to break through stiction and recover no-friction performance.

The control architecture is hybrid:
    u_total = u_LQR + alpha * u_RL
where u_LQR is computed internally and u_RL is the agent's action.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from simulation.config import (
    PHYSICAL_PARAMS,
    ENV_CONFIG,
    REWARD_CONFIG,
    FrictionParams,
    ChallengeConfig,
)


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

    Physical parameters are loaded from simulation.config.PHYSICAL_PARAMS.
    Challenge configurations are loaded from simulation.config.ChallengeConfig.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        dt: float = None,
        max_voltage: float = None,
        residual_scale: float = 1.0,
        friction_params: Optional[Dict[str, float]] = None,
        domain_randomization: bool = False,
        randomization_factor: float = 0.1,
        lqr_gain_scale: float = None,
        observation_noise_std: float = None,
        disturbance_std: float = None,
        challenge_config: Optional[ChallengeConfig] = None,
        render_mode: Optional[str] = None,
        theta_gate_threshold: float = 0.05,
    ):
        """
        Initialize the environment.

        Args:
            dt: Control timestep (default from ENV_CONFIG)
            max_voltage: Maximum motor voltage (default from PHYSICAL_PARAMS)
            residual_scale: Scaling factor for RL action
            friction_params: Dict with Ts, Tc, vs, sigma (or use challenge_config)
            domain_randomization: Whether to randomize parameters each episode
            randomization_factor: ±factor variation for randomization
            lqr_gain_scale: LQR gain scaling (or use challenge_config)
            observation_noise_std: Noise on theta observation (or use challenge_config)
            disturbance_std: External disturbance on theta_dot (or use challenge_config)
            challenge_config: ChallengeConfig object (overrides individual params)
            render_mode: Rendering mode
        """
        super().__init__()

        # Use challenge_config if provided, otherwise use individual params or defaults
        if challenge_config is not None:
            self.lqr_gain_scale = challenge_config.lqr_gain_scale
            self.observation_noise_std = challenge_config.observation_noise_std
            self.disturbance_std = challenge_config.disturbance_std
            self.friction_params = challenge_config.friction.to_dict()
        else:
            self.lqr_gain_scale = lqr_gain_scale if lqr_gain_scale is not None else 1.0
            self.observation_noise_std = observation_noise_std if observation_noise_std is not None else 0.0
            self.disturbance_std = disturbance_std if disturbance_std is not None else 0.0
            if friction_params is not None:
                self.friction_params = friction_params
            else:
                self.friction_params = FrictionParams.research_friction().to_dict()

        # Store base friction for domain randomization
        self._base_friction_Ts = self.friction_params["Ts"]

        # Environment parameters
        self.dt = dt if dt is not None else ENV_CONFIG.dt
        self.max_voltage = max_voltage if max_voltage is not None else PHYSICAL_PARAMS.max_voltage
        self.residual_scale = residual_scale
        self.domain_randomization = domain_randomization
        self.randomization_factor = randomization_factor
        self.render_mode = render_mode
        self.theta_gate_threshold = theta_gate_threshold

        # Physical parameters (from config)
        self.g = PHYSICAL_PARAMS.g
        self.Mh = PHYSICAL_PARAMS.Mh
        self.Mr = PHYSICAL_PARAMS.Mr
        self.L = PHYSICAL_PARAMS.L
        self.d = PHYSICAL_PARAMS.d
        self.r = PHYSICAL_PARAMS.r
        self.r_in = PHYSICAL_PARAMS.r_in
        self.tau_stall = PHYSICAL_PARAMS.tau_stall
        self.i_stall = PHYSICAL_PARAMS.i_stall
        self.Rm = PHYSICAL_PARAMS.Rm
        self.Kt = PHYSICAL_PARAMS.Kt
        self.Jh = PHYSICAL_PARAMS.Jh
        self.Jr = PHYSICAL_PARAMS.Jr
        self.lambda_damping = PHYSICAL_PARAMS.lambda_damping
        self.b1 = PHYSICAL_PARAMS.b1
        self.b2 = PHYSICAL_PARAMS.b2
        self.Kv = PHYSICAL_PARAMS.Kv  # Back-EMF constant from no-load motor specs

        # LQR gains - computed for INVERTED pendulum (theta=0 upright)
        base_K = np.array([-45.0, 0.0, -5.2, -0.62])
        self.K = self.lqr_gain_scale * base_K

        # State limits for observation space (5D: state + prev_action_normalized)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -ENV_CONFIG.max_theta_dot, -ENV_CONFIG.max_alpha_dot, -1.0]),
            high=np.array([np.pi, np.inf, ENV_CONFIG.max_theta_dot, ENV_CONFIG.max_alpha_dot, 1.0]),
            dtype=np.float32,
        )

        # Action space: residual control signal [-1, 1], scaled by residual_scale
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # State variables
        self.state = None
        self.steps = 0
        self.max_steps = ENV_CONFIG.max_steps
        self.prev_action = 0.0  # Previous RL action (volts) for smoothness penalty

        # For rendering
        self.screen = None
        self.clock = None

    def _stribeck_friction(self, omega: float, applied_torque: float = 0.0) -> float:
        """
        Compute Stribeck friction torque as a function of angular velocity.

        Standard Stribeck model:
            F = (Tc + (Ts-Tc)*exp(-|ω|/vs))*sign(ω) + σ*ω

        Args:
            omega: Angular velocity (rad/s)
            applied_torque: Motor torque being applied (used for stiction calculation)

        Returns:
            Friction torque (Nm) - opposes motion
        """
        Ts = self.friction_params["Ts"]
        Tc = self.friction_params["Tc"]
        vs = self.friction_params["vs"]
        sigma = self.friction_params["sigma"]

        # Handle near-zero velocity case (stiction region)
        stiction_threshold = 0.01  # rad/s - very small threshold
        if abs(omega) < stiction_threshold:
            # At rest: friction opposes applied torque up to Ts
            if abs(applied_torque) <= Ts:
                return applied_torque  # Static friction matches applied torque
            else:
                return Ts * np.sign(applied_torque)  # Breakaway

        # Standard Stribeck friction model
        sign_omega = np.sign(omega)

        # Stribeck curve: high at low speeds, drops to Tc at high speeds
        stribeck_term = Tc + (Ts - Tc) * np.exp(-abs(omega) / vs)

        # Viscous friction (linear with velocity)
        viscous_term = sigma * omega

        # Total friction opposes motion direction
        return stribeck_term * sign_omega + viscous_term

    def _lqr_control(self, state: np.ndarray) -> float:
        """
        Compute LQR control signal with optional observation noise.

        u_LQR = -K · x_noisy

        Args:
            state: [theta, alpha, theta_dot, alpha_dot]

        Returns:
            Control voltage (V)
        """
        # Add observation noise to simulate sensor errors
        if self.observation_noise_std > 0:
            noisy_state = state.copy()
            noisy_state[0] += self.np_random.normal(0, self.observation_noise_std)  # theta noise
            noisy_state[2] += self.np_random.normal(0, self.observation_noise_std * 10)  # theta_dot noise
            return -np.dot(self.K, noisy_state)
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

        # Compute state-space matrix elements for INVERTED pendulum
        #
        # IMPORTANT: The MATLAB model uses theta=0 at the DOWNWARD position,
        # but our simulation uses theta=0 at the UPRIGHT position (like firmware).
        #
        # For an INVERTED pendulum (theta=0 upright):
        # - Gravity is DESTABILIZING: small theta → gravity accelerates fall
        # - So l_31 must be POSITIVE (opposite sign from regular pendulum)
        #
        # The coupling to wheel (l_41) also changes sign.

        # Pendulum equation coefficients
        l_31 = +MhgL_MrgL / MrL2_Jh  # POSITIVE for inverted pendulum (destabilizing)
        l_33 = -self.b1 / MrL2_Jh
        l_34 = (self.b2 + self.Kt * self.Kv / self.Rm) / MrL2_Jh
        l_3 = -(12.0 * self.Kt) / (self.Rm * MrL2_Jh)

        # Wheel equation coefficients
        l_41 = -MhgL_MrgL / MrL2_Jh  # NEGATIVE (reaction to pendulum gravity)
        l_43 = self.b1 / MrL2_Jh
        l_44 = -((MrL2_Jh + self.Jr) * (self.b2 + self.Kt * self.Kv / self.Rm)) / (self.Jr * MrL2_Jh)
        l_4 = (12.0 * self.Kt * (MrL2_Jh + self.Jr)) / (self.Rm * self.Jr * MrL2_Jh)

        # Motor torque including back-EMF: I = (V - Kv*ω)/Rm, τ = Kt*I
        # At high ω, back-EMF reduces current → less torque → natural speed limit
        motor_torque = self.Kt * (12.0 * u_normalized - self.Kv * alpha_dot) / self.Rm

        # Friction torque on wheel (RESEARCH: added Stribeck friction with stiction)
        tau_friction = self._stribeck_friction(alpha_dot, motor_torque)

        # Friction coupling coefficients (from inverse mass matrix derivation)
        #
        # The bearing friction between wheel and pendulum body creates:
        #   - Torque on wheel: -tau_friction (opposes wheel motion)
        #   - Reaction on pendulum: +tau_friction (Newton's third law at bearing)
        #
        # After decoupling through the inverse mass matrix M^(-1):
        #   theta_ddot += +tau_friction / MrL2_Jh
        #   alpha_ddot += -tau_friction * (MrL2_Jh + Jr) / (Jr * MrL2_Jh)
        #
        # Key physical consequence: when friction cancels motor torque (stiction),
        # the net effect on BOTH pendulum and wheel is zero. The motor cannot
        # control the pendulum if the wheel is stuck.
        friction_coeff_theta = 1.0 / MrL2_Jh
        friction_coeff_alpha = (MrL2_Jh + self.Jr) / (self.Jr * MrL2_Jh)

        # State derivatives (non-linear version using sin(theta) instead of linearized theta)
        theta_ddot = (
            l_31 * np.sin(theta) +  # Non-linear gravity term
            l_33 * theta_dot +
            l_34 * alpha_dot +
            l_3 * u_normalized +  # Motor torque effect on pendulum
            tau_friction * friction_coeff_theta  # Friction reaction on pendulum (from bearing)
        )

        alpha_ddot = (
            l_41 * np.sin(theta) +  # Non-linear gravity coupling
            l_43 * theta_dot +
            l_44 * alpha_dot +
            l_4 * u_normalized -  # Motor torque effect on wheel
            tau_friction * friction_coeff_alpha  # Friction on wheel (with correct coupling)
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

    def _get_obs(self) -> np.ndarray:
        """Build 5D observation: [theta, alpha, theta_dot, alpha_dot, prev_action_normalized]."""
        prev_normalized = self.prev_action / self.residual_scale if self.residual_scale != 0 else 0.0
        return np.append(self.state, prev_normalized).astype(np.float32)

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
        theta_init = self.np_random.uniform(*ENV_CONFIG.theta_init_range)
        alpha_init = 0.0
        theta_dot_init = self.np_random.uniform(*ENV_CONFIG.theta_dot_init_range)
        alpha_dot_init = 0.0

        self.state = np.array([theta_init, alpha_init, theta_dot_init, alpha_dot_init])
        self.steps = 0
        self.prev_action = 0.0

        # Apply domain randomization if enabled
        if self.domain_randomization:
            self._randomize_parameters()

        return self._get_obs(), {}

    def _randomize_parameters(self):
        """Apply domain randomization to physical parameters for robust sim-to-real transfer."""
        factor = self.randomization_factor

        # Randomize masses
        self.Mh = PHYSICAL_PARAMS.Mh * self.np_random.uniform(1 - factor, 1 + factor)
        self.Mr = PHYSICAL_PARAMS.Mr * self.np_random.uniform(1 - factor, 1 + factor)

        # Randomize lengths
        self.L = PHYSICAL_PARAMS.L * self.np_random.uniform(1 - factor, 1 + factor)
        self.d = PHYSICAL_PARAMS.d * self.np_random.uniform(1 - factor, 1 + factor)

        # Recalculate inertias
        self.Jh = (1/3) * self.Mh * self.L**2
        self.Jr = (1/2) * self.Mr * (self.r**2 + self.r_in**2)

        # Randomize friction parameters around the configured baseline
        # Uses _base_friction_Ts stored at init, randomized by ±15%
        self.friction_params["Ts"] = self._base_friction_Ts * self.np_random.uniform(0.85, 1.15)
        self.friction_params["Tc"] = self.friction_params["Ts"] * 0.6
        self.friction_params["vs"] = 0.1 * self.np_random.uniform(0.9, 1.1)
        self.friction_params["sigma"] = self.friction_params["Ts"] * 0.3

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

        # Angle-gated RL: fade residual to zero near upright to eliminate steady-state chatter
        gate = min(1.0, abs(self.state[0]) / self.theta_gate_threshold)
        u_RL = gate * u_RL

        # Compute LQR baseline control (returns voltage in range [-12V, +12V])
        u_LQR = self._lqr_control(self.state)

        # Total control signal (hybrid control)
        u_total = u_LQR + u_RL

        # Saturate voltage to physical limits
        u_total = np.clip(u_total, -self.max_voltage, self.max_voltage)

        # Normalize control signal to [-1, 1] for dynamics
        # The dynamics equations expect normalized input (will be scaled by 12V internally)
        u_normalized = u_total / self.max_voltage

        # Integrate dynamics using RK4 with sub-stepping for numerical stability.
        # Stribeck friction creates stiff dynamics (large coupling coefficients),
        # so we use multiple smaller steps within each control period.
        n_substeps = 10
        sub_dt = self.dt / n_substeps
        for _ in range(n_substeps):
            k1 = self._dynamics(self.state, u_normalized)
            k2 = self._dynamics(self.state + 0.5 * sub_dt * k1, u_normalized)
            k3 = self._dynamics(self.state + 0.5 * sub_dt * k2, u_normalized)
            k4 = self._dynamics(self.state + sub_dt * k3, u_normalized)
            self.state = self.state + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Add external disturbance (random push on pendulum)
        if self.disturbance_std > 0:
            self.state[2] += self.np_random.normal(0, self.disturbance_std)

        # Normalize pendulum angle to [-pi, pi]
        self.state[0] = self._normalize_angle(self.state[0])

        # Compute reward (penalize deviation from ideal behavior)
        reward = self._compute_reward(self.state, u_RL, self.prev_action)
        self.prev_action = u_RL

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
            "gate": gate,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, state: np.ndarray, u_RL: float, prev_u_RL: float) -> float:
        """
        Reward function balancing stabilization, control effort, and smoothness.

        Uses weights from REWARD_CONFIG.

        Args:
            state: Current state [theta, alpha, theta_dot, alpha_dot]
            u_RL: Current residual RL control signal (volts)
            prev_u_RL: Previous step's RL control signal (volts)

        Returns:
            reward: Scalar reward
        """
        theta, _, theta_dot, alpha_dot = state

        # Penalize angle deviation from upright (theta = 0)
        angle_cost = REWARD_CONFIG.angle_weight * theta**2

        # Penalize angular velocities
        velocity_cost = REWARD_CONFIG.velocity_weight * theta_dot**2
        velocity_cost += REWARD_CONFIG.wheel_velocity_weight * alpha_dot**2

        # Control effort penalty
        control_cost = REWARD_CONFIG.control_weight * u_RL**2

        # Smoothness penalty: penalize rapid action changes (kills bang-bang chatter)
        smoothness_cost = REWARD_CONFIG.smoothness_weight * (u_RL - prev_u_RL)**2

        # Total reward (negative cost)
        reward = -(angle_cost + velocity_cost + control_cost + smoothness_cost)

        # Bonus for staying upright
        if abs(theta) < REWARD_CONFIG.upright_threshold:
            reward += REWARD_CONFIG.upright_bonus

        return reward

    def _is_terminated(self, state: np.ndarray) -> bool:
        """
        Check if episode should terminate due to failure.

        Uses limits from ENV_CONFIG.

        Args:
            state: Current state

        Returns:
            True if episode should end
        """
        theta, _, theta_dot, _ = state

        # Terminate if pendulum falls too far
        if abs(theta) > ENV_CONFIG.theta_limit:
            return True

        # Terminate if spinning too fast (unstable)
        if abs(theta_dot) > ENV_CONFIG.theta_dot_limit:
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
