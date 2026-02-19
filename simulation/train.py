"""
Training script for Residual RL Stiction Compensation

This script trains a PPO agent to compensate for Stribeck friction (stiction)
that degrades optimal LQR performance on a reaction wheel pendulum.

RESEARCH CONTEXT:
- Optimal LQR (scale=1.0) without friction: 0.78° RMS
- With Stribeck friction (Ts=0.15 Nm): ~1.19° RMS (stiction dead zone)
- RL with 4V authority overcomes stiction, recovering no-friction performance

The hybrid control law is: u_total = u_LQR + α * u_RL
where u_RL is the learned stiction compensation and α is the residual_scale.

Why this is compelling:
1. Genuine problem: Stiction dead zones degrade optimal LQR
2. LQR still essential: Handles stabilization (the hard part)
3. RL role is clear: Provides supplemental torque to overcome stiction
4. Practical benefit: Restores performance without modifying LQR gains

Expected Results:
- No-friction LQR: 0.78° RMS (target performance)
- Friction LQR alone: ~1.19° RMS (degraded by stiction)
- Hybrid (LQR+RL): <0.9° RMS (compensates stiction)

Usage:
    python -m simulation.train --timesteps 500000 --save_path models/ppo_residual
"""

import argparse
import os

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from simulation.envs import ReactionWheelEnv
from simulation.plotting_callback import PlottingCallback
from simulation.config import ChallengeConfig, TRAINING_CONFIG


def get_device(force_cpu: bool = False) -> str:
    """
    Automatically detect and return the best available device.

    Priority: CUDA > MPS > CPU

    Args:
        force_cpu: If True, use CPU regardless of GPU availability

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if force_cpu:
        print("Forcing CPU usage (--cpu flag)")
        return "cpu"

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) on Apple Silicon")
    else:
        device = "cpu"
        print("Using CPU (no GPU acceleration available)")

    return device


def make_env(
    rank: int,
    seed: int = 0,
    residual_scale: float = TRAINING_CONFIG.residual_scale,
    domain_randomization: bool = TRAINING_CONFIG.domain_randomization,
    challenge_config: ChallengeConfig = None,
):
    """
    Create a single environment instance.

    Args:
        rank: Index of the environment
        seed: Random seed
        residual_scale: Scaling factor for residual action
        domain_randomization: Whether to use domain randomization
        challenge_config: Challenge configuration (default: friction_compensation)

    Returns:
        Callable that creates the environment
    """
    if challenge_config is None:
        challenge_config = ChallengeConfig.friction_compensation()

    def _init():
        env = ReactionWheelEnv(
            residual_scale=residual_scale,
            domain_randomization=domain_randomization,
            randomization_factor=TRAINING_CONFIG.randomization_factor,
            challenge_config=challenge_config,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    total_timesteps: int = TRAINING_CONFIG.total_timesteps,
    n_envs: int = TRAINING_CONFIG.n_envs,
    learning_rate: float = TRAINING_CONFIG.learning_rate,
    residual_scale: float = TRAINING_CONFIG.residual_scale,
    domain_randomization: bool = TRAINING_CONFIG.domain_randomization,
    save_path: str = "models/ppo_residual",
    tensorboard_log: str = "./logs/",
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    device: str = "auto",
):
    """
    Train PPO agent to improve LQR performance under challenging conditions.

    The RL agent learns a residual control policy. The hybrid control is:
        u_total = u_LQR + residual_scale * u_RL

    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        residual_scale: Scaling factor for residual action (higher = more RL authority)
        domain_randomization: Whether to use domain randomization
        save_path: Path to save trained model
        tensorboard_log: Path for tensorboard logs
        eval_freq: Frequency of evaluation episodes
        checkpoint_freq: Frequency of checkpoint saves
        device: Device to use ('auto', 'cuda', 'mps', or 'cpu')
    """
    # Get challenge config - stiction compensation scenario
    challenge_config = ChallengeConfig.friction_compensation()

    print("=" * 60)
    print("Training Residual RL for Stiction Compensation")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Residual scale: {residual_scale}V (stiction compensation authority)")
    print(f"Domain randomization: {domain_randomization}")
    print(f"Device: {device}")
    print()
    print("LQR Configuration (optimal, with Stribeck friction):")
    print(f"  - Gain scale: {challenge_config.lqr_gain_scale} (optimal)")
    print(f"  - Friction Ts: {challenge_config.friction.Ts} Nm (stiction)")
    print()
    print("RESEARCH OBJECTIVE: Compensate stiction dead zone")
    print("  - No-friction LQR: 0.78 deg RMS (target)")
    print("  - Friction LQR alone: ~1.19 deg RMS (degraded)")
    print("  - Target: Hybrid <0.9 deg RMS (overcome stiction)")
    print("=" * 60)

    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)

    # Create vectorized training environment
    env = make_vec_env(
        lambda: make_env(
            0,
            residual_scale=residual_scale,
            domain_randomization=domain_randomization,
            challenge_config=challenge_config,
        )(),
        n_envs=n_envs,
    )

    # Wrap with normalization for stable training
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Create separate eval environment (without domain randomization for consistent evaluation)
    eval_env = make_vec_env(
        lambda: make_env(
            0,
            residual_scale=residual_scale,
            domain_randomization=False,
            challenge_config=challenge_config,
        )(),
        n_envs=1,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward for evaluation
        clip_obs=10.0,
        training=False,  # Don't update normalization stats during eval
    )

    # Create PPO model with hyperparameters from config
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=TRAINING_CONFIG.n_steps,
        batch_size=TRAINING_CONFIG.batch_size,
        n_epochs=TRAINING_CONFIG.n_epochs,
        gamma=TRAINING_CONFIG.gamma,
        gae_lambda=TRAINING_CONFIG.gae_lambda,
        clip_range=TRAINING_CONFIG.clip_range,
        ent_coef=TRAINING_CONFIG.ent_coef,
        vf_coef=TRAINING_CONFIG.vf_coef,
        max_grad_norm=TRAINING_CONFIG.max_grad_norm,
        policy_kwargs=dict(
            net_arch=dict(
                pi=list(TRAINING_CONFIG.policy_layers),
                vf=list(TRAINING_CONFIG.value_layers),
            ),
            activation_fn=nn.Tanh,  # Tanh for easier fixed-point conversion
        ),
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=device,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=save_path,
        name_prefix="ppo_checkpoint",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="logs/eval",
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Plotting callback for learning curves
    plotting_callback = PlottingCallback(
        save_path=save_path,
        plot_freq=eval_freq,  # Update plot at same frequency as evaluation
        window_size=100,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback, plotting_callback])

    # Train the model
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(save_path, "final_model")
    model.save(final_path)
    env.save(os.path.join(save_path, "vec_normalize.pkl"))

    print(f"\nTraining complete! Model saved to {final_path}")
    print("=" * 60)

    return model, env


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent to compensate for Stribeck friction on reaction wheel pendulum"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500,000)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=4.0,
        help="Max voltage for stiction compensation (default: 4.0V)",
    )
    parser.add_argument(
        "--no_domain_rand",
        action="store_true",
        help="Disable domain randomization",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/ppo_residual",
        help="Path to save model (default: models/ppo_residual)",
    )
    parser.add_argument(
        "--tensorboard_log",
        type=str,
        default="./logs/",
        help="Tensorboard log directory (default: ./logs/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training (default: auto - automatically detect best device)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (shorthand for --device cpu)",
    )

    args = parser.parse_args()

    # Determine device
    if args.cpu:
        device = "cpu"
    elif args.device == "auto":
        device = get_device(force_cpu=False)
    else:
        device = args.device

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        residual_scale=args.residual_scale,
        domain_randomization=not args.no_domain_rand,
        save_path=args.save_path,
        tensorboard_log=args.tensorboard_log,
        device=device,
    )


if __name__ == "__main__":
    main()
