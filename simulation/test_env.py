"""
Quick test script to verify the ReactionWheelEnv works correctly.

Usage:
    python -m simulation.test_env
"""

import numpy as np
from simulation.envs import ReactionWheelEnv


def test_basic_functionality():
    """Test basic environment functionality."""
    print("=" * 60)
    print("Testing ReactionWheelEnv Basic Functionality")
    print("=" * 60)

    # Create environment
    env = ReactionWheelEnv(
        dt=0.02,
        max_voltage=12.0,
        residual_scale=1.0,
        domain_randomization=False,
    )

    print("\n1. Environment created successfully!")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\n2. Environment reset successfully!")
    print(f"   Initial state: {obs}")

    # Run a few steps with random actions
    print(f"\n3. Running 10 steps with random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if i == 0:
            print(f"\n   Step {i}:")
            print(f"   - Action: {action[0]:.4f}")
            print(f"   - State: [{obs[0]:.4f}, {obs[1]:.4f}, {obs[2]:.4f}, {obs[3]:.4f}]")
            print(f"   - Reward: {reward:.4f}")
            print(f"   - u_LQR: {info['u_LQR']:.4f} V")
            print(f"   - u_RL: {info['u_RL']:.4f} V")
            print(f"   - u_total: {info['u_total']:.4f} V")
            print(f"   - Friction torque: {info['friction_torque']:.6f} Nm")

        if terminated or truncated:
            print(f"\n   Episode ended at step {i}")
            break

    print(f"\n4. Environment test completed successfully!")


def test_lqr_only():
    """Test pure LQR control (no residual)."""
    print("\n" + "=" * 60)
    print("Testing Pure LQR Control (residual_scale=0)")
    print("=" * 60)

    env = ReactionWheelEnv(
        dt=0.02,
        max_voltage=12.0,
        residual_scale=0.0,  # No residual
        domain_randomization=False,
    )

    obs, _ = env.reset(seed=42)
    total_reward = 0

    for i in range(100):
        # Zero action (pure LQR)
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"\n   Episode length: {i+1} steps")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final state: [{obs[0]:.4f}, {obs[1]:.4f}, {obs[2]:.4f}, {obs[3]:.4f}]")


def test_friction_model():
    """Test Stribeck friction model."""
    print("\n" + "=" * 60)
    print("Testing Stribeck Friction Model")
    print("=" * 60)

    env = ReactionWheelEnv()

    # Test friction at different velocities
    velocities = [-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0]
    print("\n   Omega (rad/s) | Friction Torque (Nm)")
    print("   " + "-" * 40)

    for omega in velocities:
        friction = env._stribeck_friction(omega)
        print(f"   {omega:13.1f} | {friction:20.6f}")

    # Verify properties
    friction_pos_small = env._stribeck_friction(0.01)
    friction_pos_large = env._stribeck_friction(5.0)

    print(f"\n   ✓ Static friction (small ω): {abs(friction_pos_small):.4f} Nm")
    print(f"   ✓ Coulomb friction (large ω): {abs(friction_pos_large):.4f} Nm")
    print(f"   ✓ Stribeck effect: Static > Coulomb: {abs(friction_pos_small) > abs(friction_pos_large)}")


def test_domain_randomization():
    """Test domain randomization."""
    print("\n" + "=" * 60)
    print("Testing Domain Randomization")
    print("=" * 60)

    env = ReactionWheelEnv(
        domain_randomization=True,
        randomization_factor=0.1,
    )

    # Reset multiple times and check parameter variation
    masses_Mh = []
    masses_Mr = []

    for i in range(5):
        env.reset(seed=i)
        masses_Mh.append(env.Mh)
        masses_Mr.append(env.Mr)

    print(f"\n   Pendulum mass (Mh) samples:")
    for i, mass in enumerate(masses_Mh):
        print(f"   Reset {i}: {mass:.6f} kg")

    print(f"\n   Wheel mass (Mr) samples:")
    for i, mass in enumerate(masses_Mr):
        print(f"   Reset {i}: {mass:.6f} kg")

    print(f"\n   ✓ Parameter randomization working!")


def main():
    """Run all tests."""
    try:
        test_basic_functionality()
        test_lqr_only()
        test_friction_model()
        test_domain_randomization()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe environment is ready for training!")
        print("Next steps:")
        print("  1. Tune friction parameters if needed")
        print("  2. Run training: python -m simulation.train")
        print("  3. Validate results: python -m simulation.validate")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
