"""
Plot training results from saved evaluation data.

Usage:
    python -m simulation.plot_results
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def plot_training_results(eval_path: str = "logs/eval/evaluations.npz", save_path: str = None):
    """
    Plot training results from evaluation data.

    Args:
        eval_path: Path to evaluations.npz file
        save_path: Path to save plot (if None, displays instead)
    """
    if not os.path.exists(eval_path):
        print(f"Error: Evaluation file not found at {eval_path}")
        print("Make sure you've run training first with: python -m simulation.train")
        return

    # Load evaluation data
    data = np.load(eval_path)

    # Available keys: 'timesteps', 'results', 'ep_lengths'
    timesteps = data['timesteps']
    results = data['results']  # Shape: (n_evals, n_episodes_per_eval)
    ep_lengths = data['ep_lengths']  # Shape: (n_evals, n_episodes_per_eval)

    # Compute mean and std across evaluation episodes
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    mean_lengths = np.mean(ep_lengths, axis=1)
    std_lengths = np.std(ep_lengths, axis=1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Episode rewards
    ax1.plot(timesteps, mean_rewards, color='blue', linewidth=2, label='Mean Reward')
    ax1.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        color='blue',
        label='± 1 std'
    )

    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Training Progress - Episode Rewards', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Episode lengths
    ax2.plot(timesteps, mean_lengths, color='green', linewidth=2, label='Mean Episode Length')
    ax2.fill_between(
        timesteps,
        mean_lengths - std_lengths,
        mean_lengths + std_lengths,
        alpha=0.3,
        color='green',
        label='± 1 std'
    )

    ax2.set_xlabel('Timesteps', fontsize=12)
    ax2.set_ylabel('Episode Length (steps)', fontsize=12)
    ax2.set_title('Episode Lengths Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    final_reward = mean_rewards[-1]
    final_length = mean_lengths[-1]
    max_reward = np.max(mean_rewards)
    max_reward_step = timesteps[np.argmax(mean_rewards)]

    info_text = (
        f"Total Evaluations: {len(timesteps)}\n"
        f"Total Timesteps: {timesteps[-1]:,}\n"
        f"Final Mean Reward: {final_reward:.2f} ± {std_rewards[-1]:.2f}\n"
        f"Final Mean Length: {final_length:.1f} ± {std_lengths[-1]:.1f}\n"
        f"Best Mean Reward: {max_reward:.2f} @ {max_reward_step:,} steps"
    )
    fig.text(
        0.02, 0.02, info_text,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total timesteps: {timesteps[-1]:,}")
    print(f"Number of evaluations: {len(timesteps)}")
    print(f"\nFinal Performance:")
    print(f"  Mean reward: {final_reward:.2f} ± {std_rewards[-1]:.2f}")
    print(f"  Mean episode length: {final_length:.1f} ± {std_lengths[-1]:.1f} steps")
    print(f"\nBest Performance:")
    print(f"  Max mean reward: {max_reward:.2f}")
    print(f"  Achieved at: {max_reward_step:,} timesteps")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument(
        "--eval_path",
        type=str,
        default="logs/eval/evaluations.npz",
        help="Path to evaluations.npz file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/ppo_residual/training_results.png",
        help="Path to save plot (default: models/ppo_residual/training_results.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot instead of saving",
    )

    args = parser.parse_args()

    save_path = None if args.show else args.save_path
    plot_training_results(args.eval_path, save_path)


if __name__ == "__main__":
    main()
