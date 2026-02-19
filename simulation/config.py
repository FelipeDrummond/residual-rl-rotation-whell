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
# Friction Parameters
# =============================================================================

@dataclass
class FrictionParams:
    """
    Stribeck friction model parameters.

    The Stribeck friction model captures the nonlinear relationship between
    friction and velocity, particularly the "stiction" effect at low velocities:

        F(ω) = (Tc + (Ts-Tc)·exp(-|ω|/vs))·sign(ω) + σ·ω

    Key insight for limit cycles:
    - At low velocities (|ω| < vs), friction is HIGH (near Ts)
    - LQR commands small corrections near equilibrium
    - If LQR command < Ts, wheel doesn't move → pendulum drifts
    - Pendulum drifts → LQR increases command → exceeds Ts → wheel breaks loose
    - Sudden release causes overshoot → LQR reverses → hits stiction again
    - Result: persistent limit cycle oscillations

    For RL to be useful:
    - Ts must be significant relative to LQR commands near equilibrium
    - vs must be small (sharp stiction-to-slip transition)
    - sigma (viscous) should be low (viscous damping helps LQR, we want to challenge it)
    """

    Ts: float = 0.08  # Static friction (stiction) - N·m
    Tc: float = 0.048  # Coulomb friction (kinetic) - N·m (0.6 * Ts)
    vs: float = 0.02   # Stribeck velocity - rad/s (SMALL for sharp transition)
    sigma: float = 0.0  # Viscous friction coefficient (0 = no viscous help for LQR)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for environment."""
        return {
            "Ts": self.Ts,
            "Tc": self.Tc,
            "vs": self.vs,
            "sigma": self.sigma,
        }

    @classmethod
    def no_friction(cls) -> "FrictionParams":
        """Create zero-friction configuration."""
        return cls(Ts=0.0, Tc=0.0, vs=0.02, sigma=0.0)

    @classmethod
    def research_friction(cls) -> "FrictionParams":
        """
        Research friction: Stribeck parameters that degrade optimal LQR.

        With corrected physics (friction coupling on both pendulum and wheel
        via inverse mass matrix), these parameters create a stiction dead zone
        where the motor torque is insufficient to move the wheel at small angles.

        At Ts=0.15 Nm, optimal LQR (scale=1.0) degrades from 0.78° to ~1.19° RMS.
        The RL agent with 4V authority can reduce the dead zone and recover performance.
        """
        return cls(Ts=0.15, Tc=0.09, vs=0.02, sigma=0.0)

    @classmethod
    def limit_cycle(cls) -> "FrictionParams":
        """
        Friction parameters tuned to cause limit cycles with optimal LQR.

        The key is:
        - Ts high enough that small LQR commands can't overcome stiction
        - vs small so the transition is sharp (creates stick-slip)
        - sigma=0 so there's no viscous damping to help LQR
        """
        return cls(Ts=0.08, Tc=0.048, vs=0.02, sigma=0.0)

    @classmethod
    def mild_friction(cls) -> "FrictionParams":
        """Mild friction - LQR handles this well, for baseline comparison."""
        return cls(Ts=0.03, Tc=0.018, vs=0.02, sigma=0.0)

    @classmethod
    def severe_friction(cls) -> "FrictionParams":
        """Severe friction - LQR struggles significantly."""
        return cls(Ts=0.12, Tc=0.072, vs=0.02, sigma=0.0)


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

    RESEARCH INSIGHT (from corrected physics model):
    Stribeck friction creates a stiction dead zone where the motor torque
    is insufficient to move the wheel at small pendulum angles. This degrades
    even optimal LQR (scale=1.0) from 0.78° to ~1.19° RMS.

    The research angle is:
    1. Optimal LQR without friction: 0.78° RMS (target performance)
    2. Optimal LQR with Stribeck friction (Ts=0.15): ~1.19° RMS (the problem)
    3. Hybrid (LQR + RL with 4V authority): <0.9° RMS (the solution)

    The RL agent learns supplemental torque to overcome stiction at small angles,
    recovering the no-friction performance level.
    """

    # LQR configuration
    lqr_gain_scale: float = 1.0

    # Sensor noise - keep minimal for clean experiments
    observation_noise_std: float = 0.0

    # External disturbances - keep zero (not the research focus)
    disturbance_std: float = 0.0

    # Friction - Stribeck friction is the research challenge
    friction: FrictionParams = field(default_factory=FrictionParams.research_friction)

    @classmethod
    def optimal_lqr_baseline(cls) -> "ChallengeConfig":
        """
        No-friction LQR baseline (scale=1.0).

        This is the target performance: what optimal LQR achieves without friction.
        Expected: ~0.78° RMS.
        """
        return cls(
            lqr_gain_scale=1.0,
            observation_noise_std=0.0,
            disturbance_std=0.0,
            friction=FrictionParams.no_friction(),
        )

    @classmethod
    def friction_compensation(cls) -> "ChallengeConfig":
        """
        PRIMARY RESEARCH SCENARIO: Stiction compensation via residual RL.

        Configuration:
        - Optimal LQR gains (scale=1.0)
        - Stribeck friction (Ts=0.15 Nm) creating stiction dead zone
        - No noise/disturbances (clean experiment)

        Expected performance:
        - No-friction LQR: 0.78° RMS (target)
        - Friction LQR alone: ~1.19° RMS (degraded by stiction)
        - Hybrid (LQR+RL, 4V authority): <0.9° RMS (compensates stiction)

        The RL agent learns supplemental torque to overcome stiction,
        recovering the no-friction performance level.
        """
        return cls(
            lqr_gain_scale=1.0,
            observation_noise_std=0.0,
            disturbance_std=0.0,
            friction=FrictionParams.research_friction(),
        )

    @classmethod
    def underdamped_lqr(cls) -> "ChallengeConfig":
        """
        Legacy scenario: Underdamped LQR (scale=0.35, no friction).

        Kept for comparison. The friction_compensation() scenario
        is now the primary research focus.
        """
        return cls(
            lqr_gain_scale=0.35,
            observation_noise_std=0.0,
            disturbance_std=0.0,
            friction=FrictionParams.no_friction(),
        )


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    PPO training configuration for stiction compensation.

    KEY INSIGHT: The RL agent learns supplemental torque to overcome
    stiction dead zones that degrade optimal LQR performance.

    The residual_scale determines the maximum RL authority.
    At 4V, the RL can overcome stiction at θ>2° vs LQR alone at θ>6.6°,
    significantly reducing the dead zone where stiction locks the wheel.

    Timesteps raised to 1M after reward retuning (control_weight + smoothness
    penalty) — the tighter reward landscape needs more exploration.
    """

    # Training parameters
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    learning_rate: float = 3e-4

    # Residual RL - sized to overcome stiction dead zone
    # Action [-1, 1] maps to [-residual_scale, +residual_scale] volts
    # 4V provides enough authority to break through Ts=0.15 Nm stiction
    residual_scale: float = 4.0

    # Domain randomization - disabled for clean stiction signal first
    domain_randomization: bool = False
    randomization_factor: float = 0.10  # ±10% variation (when enabled)

    # PPO hyperparameters
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.995  # High for long-horizon stability
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005  # Low: back-EMF eliminated constant-bias exploit, let policy converge
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture (small for ESP32 deployment)
    # Virtual damping is a simple function - small network is enough
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

    # Initial state ranges (small: focus on stiction regime near equilibrium)
    theta_init_range: tuple = (-0.1, 0.1)  # ~±6 degrees (where stiction matters)
    theta_dot_init_range: tuple = (-0.2, 0.2)  # rad/s


# =============================================================================
# Reward Configuration
# =============================================================================

@dataclass
class RewardConfig:
    """
    Reward function weights.

    Tuning rationale (after Phase 5 bang-bang failure):
    - control_weight=0.05: max cost per step = 0.05*16 = 0.8, comparable to upright_bonus=1.0
    - smoothness_weight=0.02: max chatter cost (±4V swing) = 0.02*64 = 1.28 per step
    - This creates real trade-offs: using high voltage must yield proportional angle improvement,
      and rapid oscillation between ±max is very expensive.
    """

    # Cost weights
    angle_weight: float = 1.0  # Weight for theta^2
    velocity_weight: float = 0.1  # Weight for theta_dot^2
    wheel_velocity_weight: float = 0.001  # Weight for alpha_dot^2 (back-EMF naturally limits wheel speed)
    control_weight: float = 0.005  # Weight for u_RL^2 (low: back-EMF limits wheel, let RL act)
    smoothness_weight: float = 0.005  # Weight for (u_RL_t - u_RL_{t-1})^2 (mild chatter penalty)

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

# Default challenge scenario: stiction compensation with optimal LQR
CHALLENGE_CONFIG = ChallengeConfig.friction_compensation()
