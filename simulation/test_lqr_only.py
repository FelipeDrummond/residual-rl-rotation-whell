"""
Quick test to validate LQR controller with fixed physics.

This script tests the environment with LQR-only control (no RL residual)
to verify that the physics model is now correct.
"""

import numpy as np
from simulation.envs import ReactionWheelEnv


def test_lqr_only(with_friction=False, num_episodes=5):
    """
    Test LQR-only controller.

    Args:
        with_friction: Whether to enable Stribeck friction
        num_episodes: Number of test episodes
    """
    # Create environment
    if with_friction:
        print("Testing LQR controller WITH Stribeck friction...")
        env = ReactionWheelEnv(
            dt=0.02,
            max_voltage=12.0,
            residual_scale=0.0,  # No residual (LQR only)
            domain_randomization=False,
        )
    else:
        print("Testing LQR controller WITHOUT friction (baseline)...")
        env = ReactionWheelEnv(
            dt=0.02,
            max_voltage=12.0,
            residual_scale=0.0,  # No residual (LQR only)
            domain_randomization=False,
            friction_params={"Ts": 0.0, "Tc": 0.0, "vs": 0.05, "sigma": 0.0},  # Disable friction
        )

    episode_lengths = []
    episode_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            # Zero action = LQR only
            action = np.array([0.0])
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Print first episode details
            if ep == 0 and steps % 50 == 0:
                theta, alpha, theta_dot, alpha_dot = obs
                print(f"  Step {steps:3d}: θ={theta:+.3f} rad, θ̇={theta_dot:+.3f} rad/s, "
                      f"u_LQR={info['u_LQR']:+.2f}V, reward={reward:+.2f}")

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: {steps} steps, total reward: {total_reward:.2f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"Average total reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Max episode length: {np.max(episode_lengths)} steps")
    print(f"Min episode length: {np.min(episode_lengths)} steps")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if not with_friction:
        if np.mean(episode_lengths) > 400:
            print("✅ PASS: LQR controller works well without friction")
            print("   Episode lengths reach max_steps (500)")
        elif np.mean(episode_lengths) > 100:
            print("⚠️  PARTIAL: LQR controller somewhat stable but not optimal")
        else:
            print("❌ FAIL: LQR controller still unstable - physics may still be wrong")
    else:
        if np.mean(episode_lengths) < 100:
            print("✅ EXPECTED: LQR struggles with friction (this is what RL should fix)")
        else:
            print("⚠️  UNEXPECTED: LQR handles friction well (may need stronger friction)")

    env.close()
    return np.mean(episode_lengths), np.mean(episode_rewards)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LQR CONTROLLER VALIDATION")
    print("="*60)
    print()

    # Test 1: Baseline (no friction) - should work well
    print("\n### TEST 1: Baseline (No Friction) ###\n")
    baseline_length, baseline_reward = test_lqr_only(with_friction=False, num_episodes=5)

    # Test 2: With friction - should struggle
    print("\n### TEST 2: With Stribeck Friction ###\n")
    friction_length, friction_reward = test_lqr_only(with_friction=True, num_episodes=5)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Baseline (no friction):  {baseline_length:.1f} steps, {baseline_reward:.2f} reward")
    print(f"With friction:           {friction_length:.1f} steps, {friction_reward:.2f} reward")
    print(f"Performance degradation: {(1 - friction_length/baseline_length)*100:.1f}%")
    print()
    if baseline_length > 400:
        print("✅ Physics model is correct! Ready for RL training.")
    else:
        print("❌ LQR still struggles without friction. More debugging needed.")
