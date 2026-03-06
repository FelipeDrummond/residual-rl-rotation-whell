"""
Training script for Residual RL Cogging Torque Compensation

This script trains a PPO agent to compensate for cogging torque that
degrades optimal LQR performance on a reaction wheel pendulum.

RESEARCH CONTEXT:
- Cogging torque is position-dependent: τ_cog = A·sin(N·α)
- LQR with K[1]=0 (no wheel angle feedback) structurally cannot compensate it
- RL learns α-dependent supplemental torque to recover no-cogging performance

The hybrid control law is: u_total = u_LQR + α * u_RL
where u_RL is the learned cogging compensation.

Why this is compelling:
1. Structural limitation: LQR with K[1]=0 ignores wheel position entirely
2. LQR still essential: Handles stabilization (the hard part)
3. RL role is principled: Provides position-dependent compensation LQR cannot
4. Practical benefit: Cogging varies across motors; RL adapts

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
    """Create a single environment instance."""
    if challenge_config is None:
        challenge_config = ChallengeConfig.cogging_compensation()

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
    Train PPO agent to compensate for cogging torque.

    The RL agent learns a residual control policy. The hybrid control is:
        u_total = u_LQR + residual_scale * u_RL
    """
    challenge_config = ChallengeConfig.cogging_compensation()

    print("=" * 60)
    print("Training Residual RL for Cogging Torque Compensation")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Residual scale: {residual_scale}V (cogging compensation authority)")
    print(f"Domain randomization: {domain_randomization}")
    print(f"Device: {device}")
    print()
    print("LQR Configuration (optimal, with cogging torque):")
    print(f"  - Gain scale: {challenge_config.lqr_gain_scale} (optimal)")
    print(f"  - Cogging amplitude: {challenge_config.cogging.amplitude} Nm")
    print(f"  - Cogging poles: {challenge_config.cogging.n_poles}")
    print()
    print("RESEARCH OBJECTIVE: Compensate position-dependent cogging")
    print("  - LQR has K[1]=0 → structurally blind to wheel position")
    print("  - RL learns α-dependent compensation LQR cannot provide")
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

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Create separate eval environment (without domain randomization)
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
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

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
            activation_fn=nn.Tanh,
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

    plotting_callback = PlottingCallback(
        save_path=save_path,
        plot_freq=eval_freq,
        window_size=100,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback, plotting_callback])

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
        description="Train PPO agent to compensate for cogging torque on reaction wheel pendulum"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps (default: 500,000)",
    )
    parser.add_argument(
        "--n_envs", type=int, default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--residual_scale", type=float, default=TRAINING_CONFIG.residual_scale,
        help=f"Max voltage for cogging compensation (default: {TRAINING_CONFIG.residual_scale}V)",
    )
    parser.add_argument(
        "--no_domain_rand", action="store_true",
        help="Disable domain randomization",
    )
    parser.add_argument(
        "--save_path", type=str, default="models/ppo_residual",
        help="Path to save model (default: models/ppo_residual)",
    )
    parser.add_argument(
        "--tensorboard_log", type=str, default="./logs/",
        help="Tensorboard log directory (default: ./logs/)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training (default: auto)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU usage (shorthand for --device cpu)",
    )

    args = parser.parse_args()

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
