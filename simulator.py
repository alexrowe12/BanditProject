#!/usr/bin/env python3
"""
Bandit Game Simulator

Simulates the multi-armed bandit game so you can practice using bandit.py.
Run this in one terminal, and bandit.py in another.

Usage: python3 simulator.py [--rounds 25] [--arms 4] [--seed 42]
"""

import numpy as np
import argparse
from typing import List, Tuple


def generate_arm_distribution(max_reward: int = 10) -> np.ndarray:
    """
    Generate a random categorical distribution for an arm.

    Uses a Dirichlet distribution to create varied arm qualities.
    Some arms will be clearly better than others.
    """
    # Use different concentration parameters to create variety
    # Lower alpha = more peaked distributions (arm specializes in certain rewards)
    # Higher alpha = more uniform distributions

    alpha = np.random.uniform(0.5, 3.0, size=max_reward + 1)

    # Optionally bias some arms to be better/worse
    # Shift the alpha to favor higher or lower rewards
    bias = np.random.uniform(-1, 1)  # Positive = better arm
    if bias > 0:
        # Favor higher rewards
        alpha = alpha * np.linspace(0.5, 2.0, max_reward + 1)
    else:
        # Favor lower rewards
        alpha = alpha * np.linspace(2.0, 0.5, max_reward + 1)

    # Ensure all alphas are positive
    alpha = np.maximum(alpha, 0.1)

    # Sample the actual distribution
    distribution = np.random.dirichlet(alpha)

    return distribution


def compute_expected_value(distribution: np.ndarray) -> float:
    """Compute expected value of a reward distribution."""
    rewards = np.arange(len(distribution))
    return np.sum(distribution * rewards)


def sample_reward(distribution: np.ndarray) -> int:
    """Sample a reward from the distribution."""
    rewards = np.arange(len(distribution))
    return np.random.choice(rewards, p=distribution)


def print_distribution(dist: np.ndarray, name: str):
    """Print a distribution in a readable format."""
    ev = compute_expected_value(dist)
    print(f"\n{name} (Expected Value: {ev:.2f}):")

    # Show as a simple histogram
    max_bar = 30
    for i, p in enumerate(dist):
        bar_len = int(p * max_bar)
        bar = "#" * bar_len
        print(f"  {i:2d}: {bar:<{max_bar}} ({p*100:5.1f}%)")


def run_simulation(num_rounds: int, num_arms: int, max_reward: int, seed: int = None):
    """Run the bandit game simulation."""

    if seed is not None:
        np.random.seed(seed)
        print(f"Using random seed: {seed}")

    # Generate distributions for each arm
    distributions = [generate_arm_distribution(max_reward) for _ in range(num_arms)]

    # Compute and sort arms by expected value (for end reveal)
    arm_evs = [(i, compute_expected_value(distributions[i])) for i in range(num_arms)]

    print("=" * 60)
    print("BANDIT GAME SIMULATOR")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Arms: {num_arms} (numbered 0-{num_arms - 1})")
    print(f"  Rounds: {num_rounds}")
    print(f"  Rewards: 0-{max_reward}")
    print(f"\nInstructions:")
    print("  1. Run bandit.py in another terminal")
    print("  2. Each round, I'll tell you the forbidden arm")
    print("  3. Enter the arm you want to pull (from bandit.py's recommendation)")
    print("  4. I'll give you the reward to enter into bandit.py")
    print("\nPress Enter to start...")
    input()

    # Track game history
    history: List[Tuple[int, int, int]] = []  # (arm, reward, forbidden)
    total_reward = 0

    # Main game loop
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_num}/{num_rounds}")
        print("=" * 60)

        # Randomly select forbidden arm
        forbidden = np.random.randint(0, num_arms)

        print(f"\n>>> FORBIDDEN ARM THIS ROUND: {forbidden}")
        print(f"\n    [Enter '{forbidden}' as the forbidden arm in bandit.py]")

        # Get user's arm choice
        while True:
            try:
                choice_input = input(f"\nWhich arm do you want to pull? (0-{num_arms - 1}): ").strip()
                chosen_arm = int(choice_input)

                if chosen_arm < 0 or chosen_arm >= num_arms:
                    print(f"  Please enter a number between 0 and {num_arms - 1}")
                    continue

                if chosen_arm == forbidden:
                    print(f"  Warning: Arm {chosen_arm} is FORBIDDEN this round!")
                    confirm = input("  Are you sure? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue

                break
            except ValueError:
                print("  Please enter a valid integer")
            except EOFError:
                print("\nExiting simulation.")
                return

        # Sample reward from chosen arm's distribution
        reward = sample_reward(distributions[chosen_arm])
        total_reward += reward
        history.append((chosen_arm, reward, forbidden))

        print(f"\n>>> YOUR REWARD: {reward}")
        print(f"\n    [Enter '{reward}' as your reward in bandit.py]")
        print(f"\n    Cumulative reward so far: {total_reward}")

        # Pause before next round
        if round_num < num_rounds:
            input("\nPress Enter for next round...")

    # Game over - reveal everything
    print("\n" + "=" * 60)
    print("GAME OVER - REVEALING TRUE DISTRIBUTIONS")
    print("=" * 60)

    print(f"\nYour total reward: {total_reward}")
    print(f"Average per round: {total_reward / num_rounds:.2f}")

    # Calculate theoretical maximum (always picking best allowed arm)
    theoretical_max = 0
    for arm, reward, forbidden in history:
        # Best arm that wasn't forbidden
        best_ev = max(compute_expected_value(distributions[i])
                      for i in range(num_arms) if i != forbidden)
        theoretical_max += best_ev

    print(f"\nTheoretical expected value (always best allowed arm): {theoretical_max:.1f}")
    print(f"Your efficiency: {(total_reward / theoretical_max) * 100:.1f}%")

    # Show true distributions ranked by expected value
    print("\n" + "-" * 60)
    print("TRUE ARM DISTRIBUTIONS (ranked best to worst):")
    print("-" * 60)

    sorted_arms = sorted(arm_evs, key=lambda x: x[1], reverse=True)

    for rank, (arm_idx, ev) in enumerate(sorted_arms, 1):
        quality = "BEST" if rank == 1 else ("WORST" if rank == num_arms else "")
        print_distribution(distributions[arm_idx], f"Arm {arm_idx} {quality}")

    # Summary table
    print("\n" + "-" * 60)
    print("SUMMARY TABLE:")
    print("-" * 60)
    print(f"{'Arm':<6} {'E[Value]':<12} {'Your Pulls':<12} {'Your Rewards':<12}")
    print("-" * 60)

    for arm_idx in range(num_arms):
        ev = compute_expected_value(distributions[arm_idx])
        pulls = sum(1 for a, r, f in history if a == arm_idx)
        rewards = sum(r for a, r, f in history if a == arm_idx)
        print(f"{arm_idx:<6} {ev:<12.2f} {pulls:<12} {rewards:<12}")

    print("-" * 60)
    print(f"{'TOTAL':<6} {'':<12} {len(history):<12} {total_reward:<12}")

    # Round-by-round history
    print("\n" + "-" * 60)
    print("ROUND-BY-ROUND HISTORY:")
    print("-" * 60)
    print(f"{'Round':<8} {'Forbidden':<12} {'You Chose':<12} {'Reward':<8} {'Best Allowed':<12}")
    print("-" * 60)

    for i, (arm, reward, forbidden) in enumerate(history, 1):
        # Find best allowed arm that round
        best_allowed = max(range(num_arms),
                          key=lambda x: compute_expected_value(distributions[x]) if x != forbidden else -1)
        best_ev = compute_expected_value(distributions[best_allowed])
        chose_best = "Yes" if arm == best_allowed else "No"
        print(f"{i:<8} {forbidden:<12} {arm:<12} {reward:<8} {best_allowed} (E={best_ev:.1f})")


def main():
    parser = argparse.ArgumentParser(description='Bandit Game Simulator')
    parser.add_argument('--rounds', type=int, default=25,
                        help='Number of rounds (default: 25)')
    parser.add_argument('--arms', type=int, default=4,
                        help='Number of arms (default: 4)')
    parser.add_argument('--max-reward', type=int, default=10,
                        help='Maximum reward value (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    run_simulation(
        num_rounds=args.rounds,
        num_arms=args.arms,
        max_reward=args.max_reward,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
