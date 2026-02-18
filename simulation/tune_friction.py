"""
Friction Parameter Tuning Script

This script finds friction parameters that create the "limit cycle" scenario:
- Optimal LQR stabilizes the pendulum (doesn't fall)
- But persistent oscillations remain due to Stribeck friction
- The oscillations have a characteristic stick-slip pattern

The goal is to find Ts (static friction) values where:
1. LQR survives all episodes (100% success rate)
2. But RMS error is elevated due to limit cycles
3. The oscillations are visible in time series plots

Usage:
    python -m simulation.tune_friction --plot
    python -m simulation.tune_friction --ts_range 0.05 0.15 --n_points 10
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass

from simulation.envs import ReactionWheelEnv
from simulation.config import FrictionParams, ChallengeConfig, ENV_CONFIG


@dataclass
class FrictionTestResult:
    """Results from testing a specific friction configuration."""
    Ts: float
    Tc: float
    vs: float
    sigma: float
    success_rate: float  # Fraction of episodes that didn't terminate early
    mean_rms_error: float  # Mean RMS error across successful episodes
    std_rms_error: float
    mean_oscillation_freq: float  # Estimated oscillation frequency
    limit_cycle_detected: bool  # Whether persistent oscillations were detected
    trajectories: List[np.ndarray]  # Sample trajectories for plotting


def estimate_oscillation_frequency(theta: np.ndarray, dt: float = 0.02) -> float:
    """
    Estimate the dominant oscillation frequency from theta time series.

    Uses zero-crossing detection to estimate frequency.
    """
    if len(theta) < 10:
        return 0.0

    # Count zero crossings in the second half (after transient)
    half = len(theta) // 2
    theta_steady = theta[half:]

    # Zero crossings
    crossings = np.where(np.diff(np.sign(theta_steady)))[0]
    n_crossings = len(crossings)

    if n_crossings < 2:
        return 0.0

    # Each full oscillation has 2 zero crossings
    duration = len(theta_steady) * dt
    freq = n_crossings / (2 * duration)

    return freq


def detect_limit_cycle(theta: np.ndarray, threshold: float = 0.02) -> bool:
    """
    Detect if a limit cycle (persistent oscillation) is present.

    A limit cycle is indicated by:
    1. The system doesn't converge to zero
    2. Oscillations persist in the second half of the trajectory

    Args:
        theta: Pendulum angle trajectory
        threshold: RMS threshold for limit cycle detection (rad)

    Returns:
        True if limit cycle detected
    """
    if len(theta) < 100:
        return False

    # Look at the last quarter of the trajectory
    quarter = len(theta) // 4
    theta_late = theta[-quarter:]

    # Check if oscillations persist
    rms_late = np.sqrt(np.mean(theta_late**2))
    max_late = np.max(np.abs(theta_late))

    # Limit cycle: RMS is elevated and oscillations are present
    has_oscillations = max_late > threshold
    not_converged = rms_late > threshold / 2

    return has_oscillations and not_converged


def run_friction_test(
    friction_params: FrictionParams,
    n_episodes: int = 20,
    max_steps: int = 500,
    verbose: bool = False,
) -> FrictionTestResult:
    """
    Test LQR performance with specific friction parameters.

    Args:
        friction_params: Friction configuration to test
        n_episodes: Number of test episodes
        max_steps: Maximum steps per episode
        verbose: Print progress

    Returns:
        FrictionTestResult with performance metrics
    """
    # Create environment with optimal LQR (no artificial handicaps)
    challenge_config = ChallengeConfig(
        lqr_gain_scale=1.0,
        observation_noise_std=0.0,
        disturbance_std=0.0,
        friction=friction_params,
    )

    env = ReactionWheelEnv(
        residual_scale=0.0,  # LQR only
        domain_randomization=False,
        challenge_config=challenge_config,
    )

    successes = 0
    rms_errors = []
    oscillation_freqs = []
    limit_cycles = []
    trajectories = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        theta_history = [obs[0]]

        for step in range(max_steps):
            # LQR only (no residual)
            action = np.array([0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            theta_history.append(obs[0])

            if terminated:
                break

        theta_array = np.array(theta_history)

        # Check if episode was successful (didn't terminate early)
        if not terminated or len(theta_history) >= max_steps:
            successes += 1
            rms = np.sqrt(np.mean(theta_array**2))
            rms_errors.append(rms)

            freq = estimate_oscillation_frequency(theta_array)
            oscillation_freqs.append(freq)

            lc = detect_limit_cycle(theta_array)
            limit_cycles.append(lc)

        # Store first few trajectories for plotting
        if len(trajectories) < 3:
            trajectories.append(theta_array)

    success_rate = successes / n_episodes

    return FrictionTestResult(
        Ts=friction_params.Ts,
        Tc=friction_params.Tc,
        vs=friction_params.vs,
        sigma=friction_params.sigma,
        success_rate=success_rate,
        mean_rms_error=np.mean(rms_errors) if rms_errors else float('inf'),
        std_rms_error=np.std(rms_errors) if rms_errors else 0.0,
        mean_oscillation_freq=np.mean(oscillation_freqs) if oscillation_freqs else 0.0,
        limit_cycle_detected=any(limit_cycles) if limit_cycles else False,
        trajectories=trajectories,
    )


def sweep_friction(
    ts_range: Tuple[float, float] = (0.02, 0.15),
    n_points: int = 8,
    vs: float = 0.02,
    n_episodes: int = 20,
    verbose: bool = True,
) -> List[FrictionTestResult]:
    """
    Sweep over Ts values to find the limit cycle region.

    Args:
        ts_range: (min, max) for static friction Ts
        n_points: Number of Ts values to test
        vs: Stribeck velocity (keep constant)
        n_episodes: Episodes per test
        verbose: Print progress

    Returns:
        List of FrictionTestResult for each Ts value
    """
    ts_values = np.linspace(ts_range[0], ts_range[1], n_points)
    results = []

    if verbose:
        print("=" * 70)
        print("FRICTION PARAMETER SWEEP")
        print("=" * 70)
        print(f"Testing Ts from {ts_range[0]:.3f} to {ts_range[1]:.3f} Nm")
        print(f"Stribeck velocity vs = {vs} rad/s")
        print(f"Viscous friction sigma = 0 (no viscous damping)")
        print("=" * 70)
        print(f"{'Ts (Nm)':<10} {'Success':<10} {'RMS (deg)':<12} {'Freq (Hz)':<10} {'Limit Cycle':<12}")
        print("-" * 70)

    for ts in ts_values:
        friction = FrictionParams(
            Ts=ts,
            Tc=ts * 0.6,  # Kinetic = 60% of static
            vs=vs,
            sigma=0.0,  # No viscous damping
        )

        result = run_friction_test(friction, n_episodes=n_episodes)
        results.append(result)

        if verbose:
            rms_deg = np.degrees(result.mean_rms_error)
            lc_str = "YES" if result.limit_cycle_detected else "no"
            print(f"{ts:<10.4f} {result.success_rate*100:<10.1f}% {rms_deg:<12.2f} {result.mean_oscillation_freq:<10.2f} {lc_str:<12}")

    if verbose:
        print("=" * 70)

        # Find the "sweet spot" - high success rate + limit cycle
        sweet_spots = [r for r in results if r.success_rate > 0.9 and r.limit_cycle_detected]
        if sweet_spots:
            best = min(sweet_spots, key=lambda r: r.mean_rms_error)
            print(f"\nRECOMMENDED Ts = {best.Ts:.4f} Nm")
            print(f"  - Success rate: {best.success_rate*100:.1f}%")
            print(f"  - RMS error: {np.degrees(best.mean_rms_error):.2f} degrees")
            print(f"  - Limit cycle detected: YES")
        else:
            print("\nNo clear limit cycle region found. Try adjusting ts_range or vs.")

    return results


def plot_friction_sweep(results: List[FrictionTestResult], save_path: str = None):
    """Plot friction sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Friction Parameter Sweep: Finding the Limit Cycle Region", fontsize=14, fontweight='bold')

    ts_values = [r.Ts for r in results]

    # Success rate
    ax = axes[0, 0]
    success_rates = [r.success_rate * 100 for r in results]
    ax.plot(ts_values, success_rates, 'o-', linewidth=2, markersize=8)
    ax.axhline(90, color='g', linestyle='--', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Static Friction Ts (Nm)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('LQR Stability vs Friction')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 105])

    # RMS error
    ax = axes[0, 1]
    rms_errors = [np.degrees(r.mean_rms_error) for r in results]
    rms_stds = [np.degrees(r.std_rms_error) for r in results]
    ax.errorbar(ts_values, rms_errors, yerr=rms_stds, fmt='o-', linewidth=2, markersize=8, capsize=4)
    ax.set_xlabel('Static Friction Ts (Nm)')
    ax.set_ylabel('RMS Error (degrees)')
    ax.set_title('Tracking Performance vs Friction')
    ax.grid(True, alpha=0.3)

    # Highlight limit cycle region
    for r in results:
        if r.limit_cycle_detected:
            ax.axvline(r.Ts, color='r', alpha=0.2, linewidth=10)

    # Oscillation frequency
    ax = axes[1, 0]
    freqs = [r.mean_oscillation_freq for r in results]
    ax.plot(ts_values, freqs, 'o-', linewidth=2, markersize=8, color='C2')
    ax.set_xlabel('Static Friction Ts (Nm)')
    ax.set_ylabel('Oscillation Frequency (Hz)')
    ax.set_title('Limit Cycle Frequency vs Friction')
    ax.grid(True, alpha=0.3)

    # Sample trajectories
    ax = axes[1, 1]
    dt = 0.02

    # Find a result with limit cycle for plotting
    lc_results = [r for r in results if r.limit_cycle_detected and r.trajectories]
    if lc_results:
        r = lc_results[len(lc_results)//2]  # Pick middle one
        for i, traj in enumerate(r.trajectories[:2]):
            time = np.arange(len(traj)) * dt
            ax.plot(time, np.degrees(traj), alpha=0.7, label=f'Episode {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pendulum Angle (degrees)')
        ax.set_title(f'Sample Trajectories (Ts={r.Ts:.3f} Nm)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'No limit cycles detected', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_single_friction_test(friction_params: FrictionParams, n_episodes: int = 5, max_steps: int = 500):
    """
    Run and plot a single friction configuration in detail.

    Useful for verifying limit cycle behavior.
    """
    challenge_config = ChallengeConfig(
        lqr_gain_scale=1.0,
        observation_noise_std=0.0,
        disturbance_std=0.0,
        friction=friction_params,
    )

    env = ReactionWheelEnv(
        residual_scale=0.0,
        domain_randomization=False,
        challenge_config=challenge_config,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'LQR with Stribeck Friction (Ts={friction_params.Ts:.3f} Nm, vs={friction_params.vs} rad/s)',
                 fontsize=14, fontweight='bold')

    dt = 0.02
    colors = plt.cm.viridis(np.linspace(0, 1, n_episodes))

    for ep in range(n_episodes):
        obs, _ = env.reset()

        theta_hist = [obs[0]]
        alpha_dot_hist = [obs[3]]
        u_hist = []
        friction_hist = []

        for step in range(max_steps):
            action = np.array([0.0])
            obs, reward, terminated, truncated, info = env.step(action)

            theta_hist.append(obs[0])
            alpha_dot_hist.append(obs[3])
            u_hist.append(info['u_total'])
            friction_hist.append(info['friction_torque'])

            if terminated:
                break

        time = np.arange(len(theta_hist)) * dt
        time_u = np.arange(len(u_hist)) * dt

        # Theta
        axes[0, 0].plot(time, np.degrees(theta_hist), color=colors[ep], alpha=0.7)

        # Alpha dot (wheel velocity)
        axes[0, 1].plot(time, alpha_dot_hist, color=colors[ep], alpha=0.7)

        # Control signal
        axes[1, 0].plot(time_u, u_hist, color=colors[ep], alpha=0.7)

        # Friction torque
        axes[1, 1].plot(time_u, friction_hist, color=colors[ep], alpha=0.7)

    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Pendulum Angle (degrees)')
    axes[0, 0].set_title('Pendulum Angle (Limit Cycle?)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', linestyle=':', alpha=0.5)

    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Wheel Velocity (rad/s)')
    axes[0, 1].set_title('Wheel Velocity (Stick-Slip?)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='k', linestyle=':', alpha=0.5)

    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Control Voltage (V)')
    axes[1, 0].set_title('LQR Control Signal')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='k', linestyle=':', alpha=0.5)

    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Friction Torque (Nm)')
    axes[1, 1].set_title('Stribeck Friction')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='k', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Tune friction parameters to find limit cycle region")
    parser.add_argument('--ts_min', type=float, default=0.02, help='Minimum Ts to test')
    parser.add_argument('--ts_max', type=float, default=0.15, help='Maximum Ts to test')
    parser.add_argument('--n_points', type=int, default=8, help='Number of Ts values to test')
    parser.add_argument('--vs', type=float, default=0.02, help='Stribeck velocity (rad/s)')
    parser.add_argument('--n_episodes', type=int, default=20, help='Episodes per test')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--test_single', type=float, default=None, help='Test single Ts value in detail')
    parser.add_argument('--save_plot', type=str, default=None, help='Path to save sweep plot')

    args = parser.parse_args()

    if args.test_single is not None:
        # Test a single friction value in detail
        friction = FrictionParams(
            Ts=args.test_single,
            Tc=args.test_single * 0.6,
            vs=args.vs,
            sigma=0.0,
        )
        plot_single_friction_test(friction)
    else:
        # Run sweep
        results = sweep_friction(
            ts_range=(args.ts_min, args.ts_max),
            n_points=args.n_points,
            vs=args.vs,
            n_episodes=args.n_episodes,
        )

        if args.plot or args.save_plot:
            plot_friction_sweep(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()
