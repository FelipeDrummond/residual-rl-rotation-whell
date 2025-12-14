"""
Validation script for comparing LQR vs Hybrid (LQR + Residual RL) control

This script evaluates the trained residual RL agent and compares performance
against pure LQR control on an underdamped system.

RESEARCH CONTEXT:
- Without damping, LQR struggles (~80% success, ~18° RMS error)
- The RL agent learns to provide "virtual damping"
- Target: >95% success rate, <2° RMS error

Usage:
    python -m simulation.validate --model_path models/ppo_residual/final_model --episodes 10

Generates:
    - Time series comparison plots (theta, velocities, control signals)
    - Performance metrics (RMS error, rewards, episode length)
    - Phase portraits showing stability improvement from RL damping
"""

import argparse
import os
from typing import List, Dict, Optional

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

    # Create environment with same residual_scale used during training
    env = ReactionWheelEnv(
        dt=0.02,
        max_voltage=12.0,
        residual_scale=2.0,  # Match training residual_scale
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
    lqr_no_friction_results: Optional[Dict[str, np.ndarray]] = None,
    save_path: str = "validation_plots.png",
    show_plot: bool = True,
):
    """
    Create comparison plots between LQR and Hybrid control.

    Args:
        lqr_results: Results from LQR-only evaluation (with friction)
        hybrid_results: Results from hybrid evaluation
        lqr_no_friction_results: Optional baseline results without friction
        save_path: Path to save plot
        show_plot: Whether to display the plot
    """
    print("Creating comparison plots...")

    dt = 0.02
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Control Performance Comparison: LQR vs Hybrid (LQR + Residual RL)", fontsize=14, fontweight='bold')

    # Plot first episode from each
    lqr_states = lqr_results["states"][0]
    hybrid_states = hybrid_results["states"][0]
    lqr_controls = lqr_results["controls"][0]
    hybrid_controls = hybrid_results["controls"][0]

    time_lqr = np.arange(len(lqr_states)) * dt
    time_hybrid = np.arange(len(hybrid_states)) * dt

    # Include baseline if available
    if lqr_no_friction_results is not None:
        baseline_states = lqr_no_friction_results["states"][0]
        time_baseline = np.arange(len(baseline_states)) * dt

    # Row 1: Pendulum angle and velocity
    if lqr_no_friction_results is not None:
        axes[0, 0].plot(time_baseline, baseline_states[:, 0], label="LQR (no friction)",
                        color="C2", linewidth=1.5, alpha=0.7)
    axes[0, 0].plot(time_lqr, lqr_states[:, 0], label="LQR (with friction)", color="C0", linewidth=2)
    axes[0, 0].plot(time_hybrid, hybrid_states[:, 0], label="Hybrid (LQR+RL)", color="C1", linewidth=2, linestyle="--")
    axes[0, 0].set_ylabel("Pendulum Angle θ (rad)", fontsize=11)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    axes[0, 0].set_title("Pendulum Angle")

    if lqr_no_friction_results is not None:
        axes[0, 1].plot(time_baseline, baseline_states[:, 2], label="LQR (no friction)",
                        color="C2", linewidth=1.5, alpha=0.7)
    axes[0, 1].plot(time_lqr, lqr_states[:, 2], label="LQR (with friction)", color="C0", linewidth=2)
    axes[0, 1].plot(time_hybrid, hybrid_states[:, 2], label="Hybrid (LQR+RL)", color="C1", linewidth=2, linestyle="--")
    axes[0, 1].set_ylabel("Pendulum Velocity θ̇ (rad/s)", fontsize=11)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    axes[0, 1].set_title("Pendulum Angular Velocity")

    # Row 2: Wheel velocity and control signal
    if lqr_no_friction_results is not None:
        axes[1, 0].plot(time_baseline, baseline_states[:, 3], label="LQR (no friction)",
                        color="C2", linewidth=1.5, alpha=0.7)
    axes[1, 0].plot(time_lqr, lqr_states[:, 3], label="LQR (with friction)", color="C0", linewidth=2)
    axes[1, 0].plot(time_hybrid, hybrid_states[:, 3], label="Hybrid (LQR+RL)", color="C1", linewidth=2, linestyle="--")
    axes[1, 0].set_ylabel("Wheel Velocity α̇ (rad/s)", fontsize=11)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    axes[1, 0].set_title("Wheel Angular Velocity")

    axes[1, 1].plot(time_lqr, lqr_controls, label="LQR (with friction)", color="C0", linewidth=2)
    axes[1, 1].plot(time_hybrid, hybrid_controls, label="Hybrid (LQR+RL)", color="C1", linewidth=2, linestyle="--")
    axes[1, 1].set_ylabel("Total Control Signal u (V)", fontsize=11)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    axes[1, 1].axhline(12, color='r', linestyle=':', linewidth=1, alpha=0.3, label='Saturation')
    axes[1, 1].axhline(-12, color='r', linestyle=':', linewidth=1, alpha=0.3)
    axes[1, 1].set_title("Control Voltage")

    # Row 3: Residual action and friction torque comparison
    if "residuals" in hybrid_results:
        hybrid_residuals = hybrid_results["residuals"][0]
        hybrid_frictions = hybrid_results["frictions"][0]

        # Plot residual action vs friction torque (what RL is compensating for)
        axes[2, 0].plot(time_hybrid, hybrid_residuals, label="RL Residual u_RL", color="C1", linewidth=2)
        axes[2, 0].plot(time_hybrid, hybrid_frictions, label="Friction Torque τ_f", color="C3", linewidth=1.5, alpha=0.7)
        axes[2, 0].set_ylabel("Torque/Voltage", fontsize=11)
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].legend(loc='upper right')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
        axes[2, 0].set_title("RL Residual vs Friction Torque")

    # Performance comparison bar chart
    lqr_rms = [np.sqrt(np.mean(states[:, 0]**2)) for states in lqr_results["states"]]
    hybrid_rms = [np.sqrt(np.mean(states[:, 0]**2)) for states in hybrid_results["states"]]
    lqr_lengths = [len(states) for states in lqr_results["states"]]
    hybrid_lengths = [len(states) for states in hybrid_results["states"]]

    # Create grouped bar chart
    x = np.arange(2)
    width = 0.35

    ax2 = axes[2, 1]
    ax2_twin = ax2.twinx()

    bars1 = ax2.bar(x - width/2, [np.mean(lqr_rms), np.mean(hybrid_rms)], width,
                    label='RMS Error (rad)', color=['C0', 'C1'], alpha=0.7)
    bars2 = ax2_twin.bar(x + width/2, [np.mean(lqr_lengths), np.mean(hybrid_lengths)], width,
                         label='Episode Length', color=['C0', 'C1'], alpha=0.4, hatch='//')

    ax2.set_ylabel('RMS Angle Error (rad)', fontsize=11)
    ax2_twin.set_ylabel('Episode Length (steps)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['LQR\n(with friction)', 'Hybrid\n(LQR+RL)'])
    ax2.set_title("Performance Summary")
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars1, [np.mean(lqr_rms), np.mean(hybrid_rms)]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, [np.mean(lqr_lengths), np.mean(hybrid_lengths)]):
        ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_phase_portrait(
    lqr_results: Dict[str, np.ndarray],
    hybrid_results: Dict[str, np.ndarray],
    save_path: str = "phase_portrait.png",
    show_plot: bool = True,
):
    """
    Create phase portrait plots showing system stability.

    Args:
        lqr_results: Results from LQR-only evaluation
        hybrid_results: Results from hybrid evaluation
        save_path: Path to save plot
        show_plot: Whether to display the plot
    """
    print("Creating phase portrait plots...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase Portraits: θ vs θ̇ (Stability Analysis)", fontsize=14, fontweight='bold')

    # LQR phase portrait
    for i, states in enumerate(lqr_results["states"]):
        alpha = 0.3 if i > 0 else 1.0
        lw = 1 if i > 0 else 2
        axes[0].plot(states[:, 0], states[:, 2], color='C0', alpha=alpha, linewidth=lw)
    axes[0].scatter([0], [0], color='green', s=100, zorder=5, marker='*', label='Target (upright)')
    axes[0].set_xlabel("θ (rad)", fontsize=11)
    axes[0].set_ylabel("θ̇ (rad/s)", fontsize=11)
    axes[0].set_title("LQR Control (with friction)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([-1.2, 1.2])
    axes[0].set_ylim([-10, 10])

    # Hybrid phase portrait
    for i, states in enumerate(hybrid_results["states"]):
        alpha = 0.3 if i > 0 else 1.0
        lw = 1 if i > 0 else 2
        axes[1].plot(states[:, 0], states[:, 2], color='C1', alpha=alpha, linewidth=lw)
    axes[1].scatter([0], [0], color='green', s=100, zorder=5, marker='*', label='Target (upright)')
    axes[1].set_xlabel("θ (rad)", fontsize=11)
    axes[1].set_ylabel("θ̇ (rad/s)", fontsize=11)
    axes[1].set_title("Hybrid Control (LQR + RL)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([-1.2, 1.2])
    axes[1].set_ylim([-10, 10])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Phase portrait saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


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
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_results",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display plots (only save to files)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("VALIDATION: Comparing LQR vs Hybrid Control")
    print("=" * 60)

    # 1. Evaluate LQR without friction (baseline - should work perfectly)
    print("\n[1/4] Evaluating LQR without friction (baseline)...")
    no_friction_params = {"Ts": 0.0, "Tc": 0.0, "vs": 0.05, "sigma": 0.0}
    lqr_no_friction_results = evaluate_lqr_only(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        friction_params=no_friction_params,
    )

    # 2. Evaluate LQR with friction (should struggle)
    print("\n[2/4] Evaluating LQR with Stribeck friction...")
    lqr_results = evaluate_lqr_only(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        friction_params=None,  # Uses default friction
    )

    # 3. Evaluate Hybrid control (should compensate for friction)
    if os.path.exists(args.model_path + ".zip"):
        print(f"\n[3/4] Evaluating Hybrid control (LQR + RL)...")
        hybrid_results = evaluate_hybrid(
            model_path=args.model_path,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            friction_params=None,  # Uses default friction
        )

        # Print metrics
        print("\n[4/4] Computing metrics and generating plots...")
        print_metrics(lqr_results, hybrid_results)

        # Also print baseline metrics
        baseline_lengths = [len(s) for s in lqr_no_friction_results["states"]]
        baseline_rms = [np.sqrt(np.mean(s[:, 0]**2)) for s in lqr_no_friction_results["states"]]
        print(f"\nBaseline (LQR without friction):")
        print(f"  Episode Length: {np.mean(baseline_lengths):.1f} ± {np.std(baseline_lengths):.1f} steps")
        print(f"  RMS Error: {np.mean(baseline_rms):.4f} ± {np.std(baseline_rms):.4f} rad")
        print("=" * 60)

        # Generate plots
        show = not args.no_show

        # Main comparison plot
        plot_comparison(
            lqr_results,
            hybrid_results,
            lqr_no_friction_results=lqr_no_friction_results,
            save_path=os.path.join(args.output_dir, "comparison_plots.png"),
            show_plot=show,
        )

        # Phase portrait plot
        plot_phase_portrait(
            lqr_results,
            hybrid_results,
            save_path=os.path.join(args.output_dir, "phase_portrait.png"),
            show_plot=show,
        )

        print(f"\nAll plots saved to: {args.output_dir}/")
        print("  - comparison_plots.png: Time series comparison")
        print("  - phase_portrait.png: Stability analysis")

    else:
        print(f"\nError: Model not found at {args.model_path}.zip")
        print("Please train a model first using: python -m simulation.train")
        print("\nBaseline results (LQR without friction):")
        baseline_lengths = [len(s) for s in lqr_no_friction_results["states"]]
        print(f"  Episode Length: {np.mean(baseline_lengths):.1f} steps (max: {args.max_steps})")


if __name__ == "__main__":
    main()
