"""
Validation script for comparing LQR vs Hybrid (LQR + Residual RL) control

This script evaluates the trained residual RL agent and compares performance
against pure LQR control with friction.

Usage:
    python -m simulation.validate --model_path models/ppo_residual/final_model --episodes 10
"""

import argparse
import os
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from simulation.envs import ReactionWheelEnv


def evaluate_lqr_only(
    n_episodes: int = 10,
    max_steps: int = 500,
    friction_params: Dict[str, float] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate pure LQR controller (residual_scale = 0).

    Args:
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        friction_params: Friction parameters

    Returns:
        Dictionary containing trajectories
    """
    print("Evaluating LQR-only control...")

    env = ReactionWheelEnv(
        dt=0.02,
        max_voltage=12.0,
        residual_scale=0.0,  # No residual action
        friction_params=friction_params,
        domain_randomization=False,
    )

    all_states = []
    all_controls = []
    all_rewards = []
    all_frictions = []

    for ep in range(n_episodes):
        states = []
        controls = []
        rewards = []
        frictions = []

        obs, _ = env.reset()
        for step in range(max_steps):
            # LQR only (residual action = 0)
            action = np.array([0.0])
            obs, reward, terminated, truncated, info = env.step(action)

            states.append(obs.copy())
            controls.append(info["u_total"])
            rewards.append(reward)
            frictions.append(info["friction_torque"])

            if terminated or truncated:
                break

        all_states.append(np.array(states))
        all_controls.append(np.array(controls))
        all_rewards.append(np.array(rewards))
        all_frictions.append(np.array(frictions))

    return {
        "states": all_states,
        "controls": all_controls,
        "rewards": all_rewards,
        "frictions": all_frictions,
    }


def evaluate_hybrid(
    model_path: str,
    n_episodes: int = 10,
    max_steps: int = 500,
    friction_params: Dict[str, float] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate hybrid control (LQR + Residual RL).

    Args:
        model_path: Path to trained PPO model
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        friction_params: Friction parameters

    Returns:
        Dictionary containing trajectories
    """
    print(f"Evaluating Hybrid control (LQR + RL) from {model_path}...")

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = ReactionWheelEnv(
        dt=0.02,
        max_voltage=12.0,
        residual_scale=1.0,
        friction_params=friction_params,
        domain_randomization=False,
    )

    # Try to load normalization stats if available
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        use_vec = True
    else:
        use_vec = False
        print("Warning: vec_normalize.pkl not found, using unnormalized observations")

    all_states = []
    all_controls = []
    all_residuals = []
    all_rewards = []
    all_frictions = []

    for ep in range(n_episodes):
        states = []
        controls = []
        residuals = []
        rewards = []
        frictions = []

        if use_vec:
            obs = env.reset()
        else:
            obs, _ = env.reset()

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)

            if use_vec:
                obs, reward, done, info = env.step(action)
                info = info[0]
                terminated = done[0]
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)

            if use_vec:
                current_obs = env.get_original_obs()[0]
            else:
                current_obs = obs

            states.append(current_obs.copy())
            controls.append(info["u_total"])
            residuals.append(info["u_RL"])
            rewards.append(reward if not use_vec else reward[0])
            frictions.append(info["friction_torque"])

            if terminated or truncated:
                break

        all_states.append(np.array(states))
        all_controls.append(np.array(controls))
        all_residuals.append(np.array(residuals))
        all_rewards.append(np.array(rewards))
        all_frictions.append(np.array(frictions))

    return {
        "states": all_states,
        "controls": all_controls,
        "residuals": all_residuals,
        "rewards": all_rewards,
        "frictions": all_frictions,
    }


def plot_comparison(
    lqr_results: Dict[str, np.ndarray],
    hybrid_results: Dict[str, np.ndarray],
    save_path: str = "validation_plots.png",
):
    """
    Create comparison plots between LQR and Hybrid control.

    Args:
        lqr_results: Results from LQR-only evaluation
        hybrid_results: Results from hybrid evaluation
        save_path: Path to save plot
    """
    print("Creating comparison plots...")

    dt = 0.02
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("LQR vs Hybrid (LQR + Residual RL) Control Comparison", fontsize=16)

    # Plot first episode from each
    lqr_states = lqr_results["states"][0]
    hybrid_states = hybrid_results["states"][0]
    lqr_controls = lqr_results["controls"][0]
    hybrid_controls = hybrid_results["controls"][0]

    time_lqr = np.arange(len(lqr_states)) * dt
    time_hybrid = np.arange(len(hybrid_states)) * dt

    # Row 1: Pendulum angle and velocity
    axes[0, 0].plot(time_lqr, lqr_states[:, 0], label="LQR", color="C0", linewidth=2)
    axes[0, 0].plot(time_hybrid, hybrid_states[:, 0], label="Hybrid", color="C1", linewidth=2, linestyle="--")
    axes[0, 0].set_ylabel("Pendulum Angle θ (rad)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)

    axes[0, 1].plot(time_lqr, lqr_states[:, 2], label="LQR", color="C0", linewidth=2)
    axes[0, 1].plot(time_hybrid, hybrid_states[:, 2], label="Hybrid", color="C1", linewidth=2, linestyle="--")
    axes[0, 1].set_ylabel("Pendulum Velocity θ̇ (rad/s)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)

    # Row 2: Wheel velocity and control signal
    axes[1, 0].plot(time_lqr, lqr_states[:, 3], label="LQR", color="C0", linewidth=2)
    axes[1, 0].plot(time_hybrid, hybrid_states[:, 3], label="Hybrid", color="C1", linewidth=2, linestyle="--")
    axes[1, 0].set_ylabel("Wheel Velocity α̇ (rad/s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)

    axes[1, 1].plot(time_lqr, lqr_controls, label="LQR", color="C0", linewidth=2)
    axes[1, 1].plot(time_hybrid, hybrid_controls, label="Hybrid", color="C1", linewidth=2, linestyle="--")
    axes[1, 1].set_ylabel("Total Control Signal (V)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)

    # Row 3: Residual action and performance metrics
    if "residuals" in hybrid_results:
        hybrid_residuals = hybrid_results["residuals"][0]
        axes[2, 0].plot(time_hybrid, hybrid_residuals, label="Residual u_RL", color="C2", linewidth=2)
        axes[2, 0].set_ylabel("Residual Action (V)")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)

    # Performance comparison: RMS error
    lqr_rms = [np.sqrt(np.mean(states[:, 0]**2)) for states in lqr_results["states"]]
    hybrid_rms = [np.sqrt(np.mean(states[:, 0]**2)) for states in hybrid_results["states"]]

    axes[2, 1].bar(["LQR", "Hybrid"], [np.mean(lqr_rms), np.mean(hybrid_rms)],
                   color=["C0", "C1"], alpha=0.7)
    axes[2, 1].set_ylabel("Mean RMS Angle Error (rad)")
    axes[2, 1].set_title("Performance Comparison")
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {save_path}")
    plt.show()


def print_metrics(lqr_results: Dict, hybrid_results: Dict):
    """Print performance metrics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    # RMS angle error
    lqr_rms = [np.sqrt(np.mean(states[:, 0]**2)) for states in lqr_results["states"]]
    hybrid_rms = [np.sqrt(np.mean(states[:, 0]**2)) for states in hybrid_results["states"]]

    print(f"\nRMS Angle Error:")
    print(f"  LQR:    {np.mean(lqr_rms):.4f} ± {np.std(lqr_rms):.4f} rad")
    print(f"  Hybrid: {np.mean(hybrid_rms):.4f} ± {np.std(hybrid_rms):.4f} rad")
    print(f"  Improvement: {(1 - np.mean(hybrid_rms)/np.mean(lqr_rms))*100:.1f}%")

    # Total reward
    lqr_reward = [np.sum(rewards) for rewards in lqr_results["rewards"]]
    hybrid_reward = [np.sum(rewards) for rewards in hybrid_results["rewards"]]

    print(f"\nTotal Reward:")
    print(f"  LQR:    {np.mean(lqr_reward):.2f} ± {np.std(lqr_reward):.2f}")
    print(f"  Hybrid: {np.mean(hybrid_reward):.2f} ± {np.std(hybrid_reward):.2f}")

    # Episode length
    lqr_length = [len(states) for states in lqr_results["states"]]
    hybrid_length = [len(states) for states in hybrid_results["states"]]

    print(f"\nEpisode Length:")
    print(f"  LQR:    {np.mean(lqr_length):.1f} ± {np.std(lqr_length):.1f} steps")
    print(f"  Hybrid: {np.mean(hybrid_length):.1f} ± {np.std(hybrid_length):.1f} steps")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate and compare LQR vs Hybrid control"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ppo_residual/final_model",
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--save_plots",
        type=str,
        default="validation_plots.png",
        help="Path to save plots",
    )

    args = parser.parse_args()

    # Evaluate LQR only
    lqr_results = evaluate_lqr_only(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
    )

    # Evaluate Hybrid control
    if os.path.exists(args.model_path + ".zip"):
        hybrid_results = evaluate_hybrid(
            model_path=args.model_path,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
        )

        # Print metrics
        print_metrics(lqr_results, hybrid_results)

        # Plot comparison
        plot_comparison(lqr_results, hybrid_results, save_path=args.save_plots)
    else:
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using: python -m simulation.train")


if __name__ == "__main__":
    main()
