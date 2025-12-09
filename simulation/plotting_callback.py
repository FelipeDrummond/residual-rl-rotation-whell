"""
Custom callback for plotting learning curves during training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the learning curve (episode rewards) during training.

    Creates and updates a plot showing:
    - Episode rewards over time
    - Moving average of rewards
    - Episode lengths

    The plot is saved to a file and updated periodically during training.
    """

    def __init__(
        self,
        save_path: str,
        plot_freq: int = 10000,
        window_size: int = 100,
        verbose: int = 0,
    ):
        """
        Initialize the plotting callback.

        Args:
            save_path: Directory where plots will be saved
            plot_freq: Frequency (in timesteps) to update the plot
            window_size: Window size for moving average
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.plot_freq = plot_freq
        self.window_size = window_size
        self.plot_path = os.path.join(save_path, "learning_curve.png")

        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at every step. Updates plot every plot_freq steps.

        Returns:
            True to continue training
        """
        if self.n_calls % self.plot_freq == 0:
            self._plot_learning_curve()
        return True

    def _plot_learning_curve(self):
        """
        Plot the learning curve using monitor data.
        """
        try:
            # Load results from monitor files
            # The training environment saves episode data to monitor files
            results_path = os.path.join(self.save_path, "..", "..")
            if hasattr(self.training_env, 'envs'):
                # For vectorized environments, get the log dir from first env
                if hasattr(self.training_env.envs[0], 'monitor'):
                    results_path = os.path.dirname(
                        self.training_env.envs[0].monitor.filename
                    )

            # Try to load results
            try:
                df = load_results(results_path)
                if len(df) == 0:
                    return
            except:
                # If monitor files don't exist, use our own tracking
                if not hasattr(self, 'episode_rewards'):
                    return
                self._plot_from_buffer()
                return

            # Extract data
            timesteps, rewards = ts2xy(df, 'timesteps')
            episode_rewards = df['r'].values
            episode_lengths = df['l'].values

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Episode rewards
            ax1.plot(timesteps, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')

            # Calculate and plot moving average
            if len(episode_rewards) >= self.window_size:
                moving_avg = np.convolve(
                    episode_rewards,
                    np.ones(self.window_size) / self.window_size,
                    mode='valid'
                )
                moving_timesteps = timesteps[self.window_size - 1:]
                ax1.plot(
                    moving_timesteps,
                    moving_avg,
                    color='red',
                    linewidth=2,
                    label=f'Moving Average (window={self.window_size})'
                )

            ax1.set_xlabel('Timesteps', fontsize=12)
            ax1.set_ylabel('Episode Reward', fontsize=12)
            ax1.set_title('Learning Curve - Episode Rewards', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Episode lengths
            ax2.plot(timesteps, episode_lengths, alpha=0.3, color='green', label='Episode Length')

            # Calculate and plot moving average for lengths
            if len(episode_lengths) >= self.window_size:
                moving_avg_len = np.convolve(
                    episode_lengths,
                    np.ones(self.window_size) / self.window_size,
                    mode='valid'
                )
                ax2.plot(
                    moving_timesteps,
                    moving_avg_len,
                    color='orange',
                    linewidth=2,
                    label=f'Moving Average (window={self.window_size})'
                )

            ax2.set_xlabel('Timesteps', fontsize=12)
            ax2.set_ylabel('Episode Length', fontsize=12)
            ax2.set_title('Episode Lengths Over Time', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

            # Add info text
            info_text = (
                f"Total Episodes: {len(episode_rewards)}\n"
                f"Total Timesteps: {timesteps[-1] if len(timesteps) > 0 else 0}\n"
                f"Mean Reward (last {self.window_size}): "
                f"{np.mean(episode_rewards[-self.window_size:]):.2f}"
            )
            fig.text(
                0.02, 0.02, info_text,
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            if self.verbose > 0:
                print(f"Learning curve saved to {self.plot_path}")

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not create learning curve plot: {e}")

    def _plot_from_buffer(self):
        """Fallback plotting from internal buffer if monitor files unavailable."""
        if not hasattr(self, 'episode_rewards') or len(self.episode_rewards) == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        episodes = np.arange(len(self.episode_rewards))
        ax.plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')

        if len(self.episode_rewards) >= self.window_size:
            moving_avg = np.convolve(
                self.episode_rewards,
                np.ones(self.window_size) / self.window_size,
                mode='valid'
            )
            ax.plot(
                episodes[self.window_size - 1:],
                moving_avg,
                color='red',
                linewidth=2,
                label=f'Moving Average (window={self.window_size})'
            )

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Reward', fontsize=12)
        ax.set_title('Learning Curve - Episode Rewards', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()


class RewardLoggingCallback(BaseCallback):
    """
    Simpler callback that just tracks rewards for plotting at the end.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []
        self.current_length = 0

    def _on_step(self) -> bool:
        """Track rewards at each step."""
        # Add reward from current step
        self.current_rewards.append(self.locals.get('rewards', [0])[0])
        self.current_length += 1

        # Check if episode ended
        dones = self.locals.get('dones', [False])
        if dones[0]:
            # Episode finished, log total reward
            episode_reward = sum(self.current_rewards)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.current_length)

            # Reset for next episode
            self.current_rewards = []
            self.current_length = 0

        return True

    def plot_results(self, save_path: str, window_size: int = 100):
        """
        Plot the results at the end of training.

        Args:
            save_path: Path to save the plot
            window_size: Window size for moving average
        """
        if len(self.episode_rewards) == 0:
            print("No episode data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        episodes = np.arange(len(self.episode_rewards))

        # Plot 1: Episode rewards
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')

        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(
                self.episode_rewards,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax1.plot(
                episodes[window_size - 1:],
                moving_avg,
                color='red',
                linewidth=2,
                label=f'Moving Average (window={window_size})'
            )

        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Episode Reward', fontsize=12)
        ax1.set_title('Learning Curve - Episode Rewards', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Episode lengths
        ax2.plot(episodes, self.episode_lengths, alpha=0.3, color='green', label='Episode Length')

        if len(self.episode_lengths) >= window_size:
            moving_avg_len = np.convolve(
                self.episode_lengths,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax2.plot(
                episodes[window_size - 1:],
                moving_avg_len,
                color='orange',
                linewidth=2,
                label=f'Moving Average (window={window_size})'
            )

        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Episode Length', fontsize=12)
        ax2.set_title('Episode Lengths Over Time', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Add statistics
        info_text = (
            f"Total Episodes: {len(self.episode_rewards)}\n"
            f"Mean Reward (all): {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}\n"
            f"Mean Reward (last {window_size}): {np.mean(self.episode_rewards[-window_size:]):.2f}\n"
            f"Max Reward: {np.max(self.episode_rewards):.2f}\n"
            f"Min Reward: {np.min(self.episode_rewards):.2f}"
        )
        fig.text(
            0.02, 0.02, info_text,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Learning curve saved to {save_path}")
