"""
Training script for Residual RL agent on Reaction Wheel Pendulum

This script trains a PPO agent to learn a residual control policy that compensates
for extreme Stribeck friction that the LQR controller cannot handle.

RESEARCH CONTEXT:
- The LQR is robust to light/medium friction but FAILS at extreme levels (Ts >= 0.5)
- With extreme friction: LQR has 0% success rate due to stiction and non-linearity
- The RL agent learns to anticipate and overcome friction through residual control
- Target: Achieve >90% success rate where LQR completely fails

The hybrid control law is: u_total = u_LQR + α * u_RL
where u_RL is the learned residual and α is the residual_scale parameter.

Usage:
    python -m simulation.train --timesteps 500000 --save_path models/ppo_residual
"""

import argparse
import os
from datetime import datetime

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
    residual_scale: float = 1.0,
    domain_randomization: bool = True,
):
    """
    Create a single environment instance.

    Args:
        rank: Index of the environment
        seed: Random seed
        residual_scale: Scaling factor for residual action
        domain_randomization: Whether to use domain randomization

    Returns:
        Callable that creates the environment
    """
    def _init():
        env = ReactionWheelEnv(
            dt=0.02,
            max_voltage=12.0,
            residual_scale=residual_scale,
            domain_randomization=domain_randomization,
            randomization_factor=0.1,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    residual_scale: float = 2.0,  # Increased for more RL authority
    domain_randomization: bool = True,
    save_path: str = "models/ppo_residual",
    tensorboard_log: str = "./logs/",
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    device: str = "auto",
):
    """
    Train PPO agent to provide virtual damping for underdamped pendulum.

    The RL agent learns a residual control policy that compensates for the
    lack of natural damping in the system. The hybrid control is:
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
    print("=" * 60)
    print("Training Residual RL for Extreme Friction Compensation")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Residual scale: {residual_scale} (RL authority)")
    print(f"Domain randomization: {domain_randomization}")
    print(f"Device: {device}")
    print()
    print("OBJECTIVE: Learn to compensate for extreme Stribeck friction")
    print("  - LQR baseline with extreme friction: 0% success (complete failure)")
    print("  - Target: >90% success rate where LQR fails")
    print("=" * 60)

    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)

    # Create vectorized training environment
    env = make_vec_env(
        lambda: make_env(0, residual_scale=residual_scale, domain_randomization=domain_randomization)(),
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
        lambda: make_env(0, residual_scale=residual_scale, domain_randomization=False)(),
        n_envs=1,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward for evaluation
        clip_obs=10.0,
        training=False,  # Don't update normalization stats during eval
    )

    # Create PPO model
    # Hyperparameters tuned for learning damping-like behavior:
    # - Higher gamma (0.995) for long-horizon stability
    # - More n_steps for better trajectory sampling
    # - Moderate entropy to encourage exploration of damping strategies
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,  # Higher gamma for long-term stability (damping is a long-horizon task)
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # Lower entropy - damping requires consistent behavior
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),  # Small network for ESP32 deployment
            activation_fn=nn.Tanh,  # Use tanh for easier fixed-point conversion
        ),
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=device,  # Use specified device (cuda/mps/cpu)
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
        description="Train PPO agent to provide virtual damping for underdamped pendulum"
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
        default=2.0,
        help="Scaling factor for residual action - higher = more RL authority (default: 2.0)",
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
