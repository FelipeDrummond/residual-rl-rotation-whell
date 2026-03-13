"""
Reaction Wheel Inverted Pendulum Gymnasium Environment

This environment simulates a reaction wheel inverted pendulum with cogging torque,
designed for training residual RL agents to compensate for position-dependent
disturbances that LQR structurally cannot handle.

Cogging torque is position-dependent (τ = A·sin(N·α)), and since LQR uses K[1]=0
(no wheel angle feedback), it cannot compensate this disturbance. The RL agent
learns α-dependent supplemental torque to recover no-cogging performance.

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
    CoggingParams,
    ChallengeConfig,
    PhysicalParams,
)


class ReactionWheelEnv(gym.Env):
    """
    Reaction Wheel Inverted Pendulum Environment with Cogging Torque

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
        domain_randomization: bool = False,
        randomization_factor: float = 0.1,
        lqr_gain_scale: float = None,
        challenge_config: Optional[ChallengeConfig] = None,
        physical_params: Optional[PhysicalParams] = None,
        lqr_gains: Optional[Tuple[float, float, float, float]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Use challenge_config if provided, otherwise use defaults
        if challenge_config is not None:
            self.lqr_gain_scale = challenge_config.lqr_gain_scale
            self.cogging_amplitude = challenge_config.cogging.amplitude
            self.cogging_n_poles = challenge_config.cogging.n_poles
            self.cogging_phase_offset = challenge_config.cogging.phase_offset
        else:
            self.lqr_gain_scale = lqr_gain_scale if lqr_gain_scale is not None else 1.0
            cogging = CoggingParams.research_cogging()
            self.cogging_amplitude = cogging.amplitude
            self.cogging_n_poles = cogging.n_poles
            self.cogging_phase_offset = cogging.phase_offset

        # Store base cogging amplitude for domain randomization
        self._base_cogging_amplitude = self.cogging_amplitude

        # Physical parameters: use provided or default
        params = physical_params if physical_params is not None else PHYSICAL_PARAMS
        self._physical_params = params

        # Environment parameters
        self.dt = dt if dt is not None else ENV_CONFIG.dt
        self.max_voltage = max_voltage if max_voltage is not None else params.max_voltage
        self.residual_scale = residual_scale
        self.domain_randomization = domain_randomization
        self.randomization_factor = randomization_factor
        self.render_mode = render_mode

        # Physical parameters (from config)
        self.g = params.g
        self.Mh = params.Mh
        self.Mr = params.Mr
        self.L = params.L
        self.d = params.d
        self.r = params.r
        self.r_in = params.r_in
        self.tau_stall = params.tau_stall
        self.i_stall = params.i_stall
        self.Rm = params.Rm
        self.Kt = params.Kt
        self.Jh = params.Jh
        self.Jr = params.Jr
        self.lambda_damping = params.lambda_damping
        self.b1 = params.b1
        self.b2 = params.b2
        self.Kv = params.Kv

        # LQR gains - use provided or default
        if lqr_gains is not None:
            base_K = np.array(lqr_gains)
        else:
            base_K = np.array([-45.0, 0.0, -5.2, -0.62])
        self.K = self.lqr_gain_scale * base_K

        # Observation space: 4D state [theta, alpha, theta_dot, alpha_dot]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -ENV_CONFIG.max_theta_dot, -ENV_CONFIG.max_alpha_dot]),
            high=np.array([np.pi, np.inf, ENV_CONFIG.max_theta_dot, ENV_CONFIG.max_alpha_dot]),
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

        # For rendering
        self.screen = None
        self.clock = None

    def _cogging_torque(self, alpha: float) -> float:
        """
        Compute cogging torque as a function of wheel position.

        τ_cog = amplitude · sin(n_poles · α + phase_offset)

        Args:
            alpha: Wheel angle (rad)

        Returns:
            Cogging torque (Nm)
        """
        return self.cogging_amplitude * np.sin(
            self.cogging_n_poles * alpha + self.cogging_phase_offset
        )

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
        Compute state derivatives for the coupled pendulum-wheel system with cogging.

        Implements the MATLAB state-space model with non-linear extension (sin(θ))
        and added cogging torque on the wheel.

        Args:
            state: [theta, alpha, theta_dot, alpha_dot]
            u_normalized: Normalized control input (-1 to 1, scaled by 12V internally)

        Returns:
            state_dot: [theta_dot, alpha_dot, theta_ddot, alpha_ddot]
        """
        theta, alpha, theta_dot, alpha_dot = state

        # Shorthand for common terms (matching MATLAB notation)
        MrL2_Jh = self.Mr * self.L**2 + self.Jh
        MhgL_MrgL = self.Mh * self.g * self.d + self.Mr * self.g * self.L

        # Pendulum equation coefficients (INVERTED: gravity destabilizing)
        l_31 = +MhgL_MrgL / MrL2_Jh
        l_33 = -self.b1 / MrL2_Jh
        l_34 = (self.b2 + self.Kt * self.Kv / self.Rm) / MrL2_Jh
        l_3 = -(12.0 * self.Kt) / (self.Rm * MrL2_Jh)

        # Wheel equation coefficients
        l_41 = -MhgL_MrgL / MrL2_Jh
        l_43 = self.b1 / MrL2_Jh
        l_44 = -((MrL2_Jh + self.Jr) * (self.b2 + self.Kt * self.Kv / self.Rm)) / (self.Jr * MrL2_Jh)
        l_4 = (12.0 * self.Kt * (MrL2_Jh + self.Jr)) / (self.Rm * self.Jr * MrL2_Jh)

        # Cogging torque (position-dependent disturbance on wheel)
        tau_cog = self._cogging_torque(alpha)

        # Cogging coupling coefficients (from inverse mass matrix)
        # Cogging acts on wheel bearing → reaction on pendulum (Newton's 3rd law)
        cogging_coeff_theta = 1.0 / MrL2_Jh
        cogging_coeff_alpha = (MrL2_Jh + self.Jr) / (self.Jr * MrL2_Jh)

        # State derivatives (non-linear version using sin(theta))
        theta_ddot = (
            l_31 * np.sin(theta) +
            l_33 * theta_dot +
            l_34 * alpha_dot +
            l_3 * u_normalized +
            tau_cog * cogging_coeff_theta  # Cogging reaction on pendulum
        )

        alpha_ddot = (
            l_41 * np.sin(theta) +
            l_43 * theta_dot +
            l_44 * alpha_dot +
            l_4 * u_normalized -
            tau_cog * cogging_coeff_alpha  # Cogging on wheel
        )

        return np.array([theta_dot, alpha_dot, theta_ddot, alpha_ddot])

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if options and "initial_state" in options:
            theta_init, alpha_init, theta_dot_init, alpha_dot_init = options["initial_state"]
        else:
            theta_init = self.np_random.uniform(*ENV_CONFIG.theta_init_range)
            alpha_init = 0.0
            theta_dot_init = self.np_random.uniform(*ENV_CONFIG.theta_dot_init_range)
            alpha_dot_init = 0.0

        self.state = np.array([theta_init, alpha_init, theta_dot_init, alpha_dot_init])
        self.steps = 0

        if self.domain_randomization:
            self._randomize_parameters()

        return self.state.astype(np.float32), {}

    def _randomize_parameters(self):
        """Apply domain randomization to physical parameters."""
        factor = self.randomization_factor
        params = self._physical_params

        # Randomize masses
        self.Mh = params.Mh * self.np_random.uniform(1 - factor, 1 + factor)
        self.Mr = params.Mr * self.np_random.uniform(1 - factor, 1 + factor)

        # Randomize lengths
        self.L = params.L * self.np_random.uniform(1 - factor, 1 + factor)
        self.d = params.d * self.np_random.uniform(1 - factor, 1 + factor)

        # Recalculate inertias
        self.Jh = (1/3) * self.Mh * self.L**2
        self.Jr = (1/2) * self.Mr * (self.r**2 + self.r_in**2)

        # Randomize cogging amplitude ±15%
        self.cogging_amplitude = self._base_cogging_amplitude * self.np_random.uniform(0.85, 1.15)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment.

        Args:
            action: Residual control signal from RL agent (normalized to [-1, 1])

        Returns:
            observation, reward, terminated, truncated, info
        """
        u_RL = float(action[0]) * self.residual_scale

        # Compute LQR baseline control
        u_LQR = self._lqr_control(self.state)

        # Total control signal (hybrid control)
        u_total = u_LQR + u_RL

        # Saturate voltage to physical limits
        u_total = np.clip(u_total, -self.max_voltage, self.max_voltage)

        # Normalize control signal to [-1, 1] for dynamics
        u_normalized = u_total / self.max_voltage

        # RK4 integration with sub-stepping
        n_substeps = 10
        sub_dt = self.dt / n_substeps
        for _ in range(n_substeps):
            k1 = self._dynamics(self.state, u_normalized)
            k2 = self._dynamics(self.state + 0.5 * sub_dt * k1, u_normalized)
            k3 = self._dynamics(self.state + 0.5 * sub_dt * k2, u_normalized)
            k4 = self._dynamics(self.state + sub_dt * k3, u_normalized)
            self.state = self.state + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Normalize pendulum angle to [-pi, pi]
        self.state[0] = self._normalize_angle(self.state[0])

        # Compute reward
        reward = self._compute_reward(self.state, u_RL)

        # Check termination conditions
        terminated = self._is_terminated(self.state)

        self.steps += 1
        truncated = self.steps >= self.max_steps

        info = {
            "u_LQR": u_LQR,
            "u_RL": u_RL,
            "u_total": u_total,
            "cogging_torque": self._cogging_torque(self.state[1]),
        }

        return self.state.astype(np.float32), reward, terminated, truncated, info

    def _compute_reward(self, state: np.ndarray, u_RL: float) -> float:
        """
        Reward function balancing stabilization and control effort.

        Args:
            state: Current state [theta, alpha, theta_dot, alpha_dot]
            u_RL: Current residual RL control signal (volts)

        Returns:
            reward: Scalar reward
        """
        theta, _, theta_dot, alpha_dot = state

        angle_cost = REWARD_CONFIG.angle_weight * theta**2
        velocity_cost = REWARD_CONFIG.velocity_weight * theta_dot**2
        velocity_cost += REWARD_CONFIG.wheel_velocity_weight * alpha_dot**2
        control_cost = REWARD_CONFIG.control_weight * u_RL**2

        reward = -(angle_cost + velocity_cost + control_cost)

        if abs(theta) < REWARD_CONFIG.upright_threshold:
            reward += REWARD_CONFIG.upright_bonus

        return reward

    def _is_terminated(self, state: np.ndarray) -> bool:
        """Check if episode should terminate due to failure."""
        theta, _, theta_dot, _ = state

        if abs(theta) > ENV_CONFIG.theta_limit:
            return True
        if abs(theta_dot) > ENV_CONFIG.theta_dot_limit:
            return True

        return False

    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == "human":
            pass

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
