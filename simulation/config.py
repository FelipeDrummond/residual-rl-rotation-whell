"""
Configuration file for Reaction Wheel Pendulum simulation.

All physical parameters and training configurations are centralized here
for easy modification and consistency across training and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


# =============================================================================
# Physical Parameters (from MATLAB system identification)
# =============================================================================

@dataclass
class PhysicalParams:
    """Physical parameters of the reaction wheel pendulum system."""

    # Environment
    g: float = 9.81  # m/s² - gravity

    # Pendulum
    Mh: float = 0.149  # kg - pendulum mass
    L: float = 0.14298  # m - pendulum length (COM to pivot)
    d: float = 0.0987  # m - distance parameter

    # Wheel
    Mr: float = 0.144  # kg - wheel mass
    r: float = 0.1  # m - wheel outer radius
    r_in: float = 0.0911  # m - wheel inner radius

    # Motor (12V DC motor)
    max_voltage: float = 12.0  # V
    tau_stall: float = 0.3136  # Nm - stall torque
    i_stall: float = 1.8  # A - stall current
    w_noload_rpm: float = 380.0  # RPM - no-load speed
    i_noload: float = 0.1  # A - no-load current

    # Damping (from MATLAB lambda.mat)
    lambda_damping: float = 0.15060423
    b1: float = 0.0  # Pendulum damping (none)

    @property
    def Rm(self) -> float:
        """Motor resistance (Ohm)."""
        return self.max_voltage / self.i_stall

    @property
    def Kt(self) -> float:
        """Motor torque constant (Nm/A)."""
        return self.tau_stall / self.i_stall

    @property
    def Jh(self) -> float:
        """Pendulum moment of inertia (kg·m²)."""
        return (1/3) * self.Mh * self.L**2

    @property
    def Jr(self) -> float:
        """Wheel moment of inertia (kg·m²)."""
        return (1/2) * self.Mr * (self.r**2 + self.r_in**2)

    @property
    def Kv(self) -> float:
        """Back-EMF constant V/(rad/s), from no-load motor specs.

        V = Kv*ω + I_noload*Rm  →  Kv = (V - I_noload*Rm) / ω_noload
        """
        w_noload = self.w_noload_rpm * 2.0 * 3.141592653589793 / 60.0
        return (self.max_voltage - self.i_noload * self.Rm) / w_noload

    @property
    def b2(self) -> float:
        """Wheel damping coefficient (N·m·s/rad)."""
        return 2 * self.lambda_damping * (self.Jh + self.Jr)


# =============================================================================
# Cogging Torque Parameters
# =============================================================================

@dataclass
class CoggingParams:
    """
    Cogging torque model parameters.

    Cogging torque is a position-dependent disturbance from motor magnets
    interacting with stator teeth:

        τ_cog(α) = amplitude · sin(n_poles · α + phase_offset)

    Key insight for RL:
    - Cogging is a function of wheel POSITION (α), not velocity
    - LQR with K[1]=0 (no wheel angle feedback) structurally cannot compensate it
    - The RL agent must learn position-dependent compensation
    - This creates a clear, principled role for RL that LQR cannot fill
    """

    amplitude: float = 0.05  # Nm - cogging torque amplitude
    n_poles: int = 7         # Number of magnetic pole pairs
    phase_offset: float = 0.0  # rad - phase offset

    @classmethod
    def no_cogging(cls) -> "CoggingParams":
        """Create zero-cogging configuration."""
        return cls(amplitude=0.0, n_poles=7, phase_offset=0.0)

    @classmethod
    def research_cogging(cls) -> "CoggingParams":
        """
        Research cogging: parameters that degrade optimal LQR.

        At amplitude=0.05 Nm, comparable to LQR torque commands near
        equilibrium. LQR with K[1]=0 cannot compensate position-dependent
        disturbance, so performance degrades significantly.
        """
        return cls(amplitude=0.05, n_poles=7, phase_offset=0.0)


# =============================================================================
# LQR Controller Parameters
# =============================================================================

@dataclass
class LQRParams:
    """LQR controller configuration."""

    # Base gains for inverted pendulum (theta=0 upright)
    # Computed via LQR (scipy ARE) for plant WITH back-EMF (Kv=0.285)
    # Q=diag(1, 0, 0.1, 0.001), R=1, voltage-unit B matrix
    base_K: tuple = (-45.0, 0.0, -5.2, -0.62)

    # Gain scaling factor
    # 1.0 = optimal, <1.0 = undertuned (for research scenarios)
    gain_scale: float = 1.0

    @property
    def K(self) -> tuple:
        """Scaled LQR gains."""
        return tuple(k * self.gain_scale for k in self.base_K)


# =============================================================================
# Challenge Scenario Configuration
# =============================================================================

@dataclass
class ChallengeConfig:
    """
    Configuration for research challenge scenarios.

    RESEARCH INSIGHT:
    Cogging torque is position-dependent: τ_cog = A·sin(N·α). Since LQR
    uses K[1]=0 (no wheel angle feedback), it structurally cannot compensate
    this disturbance. The RL agent learns α-dependent compensation that
    LQR cannot provide.

    The research angle is:
    1. Optimal LQR without cogging: ~0.22° RMS (target performance)
    2. Optimal LQR with cogging: degraded RMS (the problem)
    3. Hybrid (LQR + RL): recovers toward baseline (the solution)
    """

    # LQR configuration
    lqr_gain_scale: float = 1.0

    # Cogging torque - the research challenge
    cogging: CoggingParams = field(default_factory=CoggingParams.research_cogging)

    @classmethod
    def optimal_lqr_baseline(cls) -> "ChallengeConfig":
        """
        No-cogging LQR baseline (scale=1.0).

        This is the target performance: what optimal LQR achieves without cogging.
        """
        return cls(
            lqr_gain_scale=1.0,
            cogging=CoggingParams.no_cogging(),
        )

    @classmethod
    def cogging_compensation(cls) -> "ChallengeConfig":
        """
        PRIMARY RESEARCH SCENARIO: Cogging compensation via residual RL.

        Configuration:
        - Optimal LQR gains (scale=1.0)
        - Cogging torque (A=0.05 Nm, 7 poles)
        - LQR with K[1]=0 cannot compensate position-dependent cogging

        The RL agent learns wheel-position-dependent compensation
        that LQR structurally cannot provide.
        """
        return cls(
            lqr_gain_scale=1.0,
            cogging=CoggingParams.research_cogging(),
        )


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    PPO training configuration for cogging torque compensation.

    KEY INSIGHT: The RL agent learns position-dependent compensation
    for cogging torque that LQR (with K[1]=0) structurally cannot provide.

    The residual_scale determines the maximum RL authority.
    At 2V, the RL has enough authority to counteract 0.05 Nm cogging.
    """

    # Training parameters
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    learning_rate: float = 3e-4

    # Residual RL - sized to counteract cogging torque
    # Action [-1, 1] maps to [-residual_scale, +residual_scale] volts
    # 2V provides enough authority to compensate 0.05 Nm cogging
    residual_scale: float = 2.0

    # Domain randomization - disabled for clean cogging signal first
    domain_randomization: bool = False
    randomization_factor: float = 0.10  # ±10% variation (when enabled)

    # PPO hyperparameters
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.995  # High for long-horizon stability
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005  # Low: let policy converge to position-dependent pattern
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture (small for ESP32 deployment)
    policy_layers: tuple = (32, 32)
    value_layers: tuple = (64, 64)


# =============================================================================
# Environment Configuration
# =============================================================================

@dataclass
class EnvConfig:
    """Environment configuration."""

    dt: float = 0.02  # 50 Hz control frequency
    max_steps: int = 500  # 10 seconds per episode

    # State limits
    max_theta_dot: float = 10.0  # rad/s
    max_alpha_dot: float = 50.0  # rad/s

    # Termination conditions
    theta_limit: float = 1.047  # ~60 degrees (pi/3)
    theta_dot_limit: float = 15.0  # rad/s

    # Initial state ranges
    theta_init_range: tuple = (-0.1, 0.1)  # ~±6 degrees
    theta_dot_init_range: tuple = (-0.2, 0.2)  # rad/s


# =============================================================================
# Reward Configuration
# =============================================================================

@dataclass
class RewardConfig:
    """
    Reward function weights.

    Simplified for cogging compensation (no smoothness term needed —
    cogging compensation is inherently smooth since it's position-dependent).
    """

    # Cost weights
    angle_weight: float = 1.0  # Weight for theta^2
    velocity_weight: float = 0.1  # Weight for theta_dot^2
    wheel_velocity_weight: float = 0.001  # Weight for alpha_dot^2
    control_weight: float = 0.005  # Weight for u_RL^2

    # Bonus
    upright_bonus: float = 1.0  # Bonus for |theta| < upright_threshold
    upright_threshold: float = 0.1  # rad (~5.7 degrees)


# =============================================================================
# Default Configurations
# =============================================================================

# Default physical parameters
PHYSICAL_PARAMS = PhysicalParams()

# Default environment config
ENV_CONFIG = EnvConfig()

# Default reward config
REWARD_CONFIG = RewardConfig()

# Default training config
TRAINING_CONFIG = TrainingConfig()

# Default challenge scenario: cogging compensation with optimal LQR
CHALLENGE_CONFIG = ChallengeConfig.cogging_compensation()
