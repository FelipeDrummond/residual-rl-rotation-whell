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
    def limit_cycle(cls) -> "FrictionParams":
        """
        Friction parameters tuned to cause limit cycles with optimal LQR.

        The key is:
        - Ts high enough that small LQR commands can't overcome stiction
        - vs small so the transition is sharp (creates stick-slip)
        - sigma=0 so there's no viscous damping to help LQR

        This creates a scenario where:
        - LQR stabilizes the pendulum (doesn't fall)
        - But persistent oscillations remain (limit cycle)
        - RL can learn to compensate and eliminate oscillations
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
    # Computed from linearized state-space model
    base_K: tuple = (-50.0, 0.0, -6.45, -0.34)

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

    RESEARCH INSIGHT (from experiments):
    The system WITHOUT friction is underdamped - LQR stabilizes but oscillates.
    Friction actually HELPS by providing natural damping.

    The compelling research angle is:
    1. Underdamped LQR (moderate gains, no friction) → oscillatory response
    2. RL learns to provide VIRTUAL DAMPING: u_RL ∝ -ω
    3. Hybrid controller achieves optimal damping without friction dependency

    This is genuine because:
    - Physical friction is unreliable (varies with temperature, wear, etc.)
    - RL learns a controllable, predictable damping function
    - LQR still handles stabilization; RL only adds damping
    """

    # LQR configuration - use moderate gains for underdamped behavior
    # Scale < 1.0 creates underdamped system that benefits from RL damping
    lqr_gain_scale: float = 0.35

    # Sensor noise - keep minimal for clean experiments
    observation_noise_std: float = 0.0

    # External disturbances - keep zero (not the research focus)
    disturbance_std: float = 0.0

    # Friction - set to ZERO to study virtual damping
    # (Physical friction would mask the RL's learned damping)
    friction: FrictionParams = field(default_factory=FrictionParams.no_friction)

    @classmethod
    def optimal_lqr_baseline(cls) -> "ChallengeConfig":
        """
        Strong LQR baseline (scale=1.0, no friction).

        This is the best-case scenario with aggressive gains.
        Shows what's achievable with well-tuned LQR alone.
        """
        return cls(
            lqr_gain_scale=1.0,
            observation_noise_std=0.0,
            disturbance_std=0.0,
            friction=FrictionParams.no_friction(),
        )

    @classmethod
    def underdamped_lqr(cls) -> "ChallengeConfig":
        """
        PRIMARY RESEARCH SCENARIO: Underdamped LQR needs virtual damping.

        Configuration:
        - Moderate LQR gains (scale=0.35) → underdamped response
        - No friction (to isolate the damping problem)
        - No noise/disturbances (clean experiment)

        Expected performance:
        - LQR alone: 100% survival, but ~2-4° RMS due to oscillations
        - Hybrid (LQR+RL): 100% survival, <1° RMS (RL provides damping)

        The RL agent learns virtual damping: u_RL(ω) ∝ -ω
        This is analogous to what friction provides, but learnable/controllable.
        """
        return cls(
            lqr_gain_scale=0.35,
            observation_noise_std=0.0,
            disturbance_std=0.0,
            friction=FrictionParams.no_friction(),
        )

    @classmethod
    def lqr_with_friction(cls) -> "ChallengeConfig":
        """
        LQR with physical friction (for comparison).

        This shows how friction naturally provides damping.
        Used to demonstrate that RL learns a similar function.
        """
        return cls(
            lqr_gain_scale=0.35,
            observation_noise_std=0.0,
            disturbance_std=0.0,
            friction=FrictionParams(Ts=0.1, Tc=0.06, vs=0.02, sigma=0.05),
        )

    @classmethod
    def varying_friction(cls) -> "ChallengeConfig":
        """
        Test robustness: train on varying friction levels.

        This tests if RL can adapt to different damping conditions.
        """
        return cls(
            lqr_gain_scale=0.35,
            observation_noise_std=0.0,
            disturbance_std=0.0,
            friction=FrictionParams(Ts=0.05, Tc=0.03, vs=0.02, sigma=0.02),
        )

    # Backward compatibility aliases
    @classmethod
    def friction_compensation(cls) -> "ChallengeConfig":
        """Alias for underdamped_lqr (the main research scenario)."""
        return cls.underdamped_lqr()

    @classmethod
    def combined_challenges(cls) -> "ChallengeConfig":
        """Alias for underdamped_lqr (backward compatibility)."""
        return cls.underdamped_lqr()

    @classmethod
    def optimal_baseline(cls) -> "ChallengeConfig":
        """Alias for optimal_lqr_baseline (backward compatibility)."""
        return cls.optimal_lqr_baseline()

    @classmethod
    def no_friction_baseline(cls) -> "ChallengeConfig":
        """Alias for optimal_lqr_baseline."""
        return cls.optimal_lqr_baseline()


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    PPO training configuration for virtual damping.

    KEY INSIGHT: The RL agent should learn to provide VIRTUAL DAMPING.
    The target function is approximately: u_RL(ω) ∝ -ω (velocity-proportional damping).

    This is similar to what physical friction provides, but:
    - Controllable and predictable
    - Doesn't depend on mechanical wear, temperature, etc.
    - Can be optimized for specific performance goals

    The residual_scale determines the maximum damping authority.
    Since the LQR is underdamped (scale=0.35), we need enough authority
    to provide significant damping (~2-3V for good damping).
    """

    # Training parameters
    total_timesteps: int = 500_000
    n_envs: int = 4
    learning_rate: float = 3e-4

    # Residual RL - sized to provide damping without overwhelming LQR
    # Action [-1, 1] maps to [-residual_scale, +residual_scale] volts
    # 2V is enough to provide significant damping
    residual_scale: float = 2.0

    # Domain randomization - vary physical parameters for robustness
    domain_randomization: bool = True
    randomization_factor: float = 0.10  # ±10% variation

    # PPO hyperparameters
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.995  # High for long-horizon stability
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01  # Moderate exploration
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

    # Initial state ranges
    theta_init_range: tuple = (-0.3, 0.3)  # ~±17 degrees
    theta_dot_init_range: tuple = (-0.5, 0.5)  # rad/s


# =============================================================================
# Reward Configuration
# =============================================================================

@dataclass
class RewardConfig:
    """Reward function weights."""

    # Cost weights
    angle_weight: float = 1.0  # Weight for theta^2
    velocity_weight: float = 0.1  # Weight for theta_dot^2
    wheel_velocity_weight: float = 0.001  # Weight for alpha_dot^2
    control_weight: float = 0.01  # Weight for u_RL^2

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

# Default challenge scenario: underdamped LQR needs virtual damping
CHALLENGE_CONFIG = ChallengeConfig.underdamped_lqr()
