#!/usr/bin/env python3
"""
Thompson Sampling with Dirichlet Priors for Multi-Armed Bandit
Enhanced with Social Learning from Other Teams' Choices

Usage: python bandit.py [--social-weight 0.3] [--num-arms 4] [--num-rounds 25]
"""

import numpy as np
import argparse
from typing import List, Optional, Tuple


class DirichletArm:
    """
    Represents a single arm with a Dirichlet prior over categorical rewards.
    Rewards are integers from 0 to max_reward (default 10).
    """

    def __init__(self, max_reward: int = 10):
        self.max_reward = max_reward
        self.num_categories = max_reward + 1  # 0 through max_reward
        # Uniform prior: alpha = [1, 1, 1, ..., 1]
        self.alpha = np.ones(self.num_categories)
        self.pull_count = 0

    def sample_expected_value(self, social_boost: np.ndarray = None) -> float:
        """
        Thompson Sampling: Draw a sample from the Dirichlet posterior,
        then compute the expected value under that sampled distribution.

        Args:
            social_boost: Optional array to add to alpha (for social learning)

        Returns:
            Expected reward value from the sampled distribution
        """
        effective_alpha = self.alpha.copy()
        if social_boost is not None:
            effective_alpha += social_boost

        # Sample a categorical distribution from Dirichlet
        sampled_probs = np.random.dirichlet(effective_alpha)

        # Compute expected value: sum(p_i * i for i in 0..max_reward)
        rewards = np.arange(self.num_categories)
        expected_value = np.sum(sampled_probs * rewards)

        return expected_value

    def update(self, reward: int):
        """Update the posterior after observing a reward."""
        if 0 <= reward <= self.max_reward:
            self.alpha[reward] += 1
            self.pull_count += 1
        else:
            raise ValueError(f"Reward must be between 0 and {self.max_reward}")

    def mean_estimate(self) -> float:
        """Return the current mean estimate (posterior mean expected value)."""
        # Posterior mean of Dirichlet is alpha / sum(alpha)
        probs = self.alpha / np.sum(self.alpha)
        rewards = np.arange(self.num_categories)
        return np.sum(probs * rewards)


class BanditGame:
    """
    Manages the multi-armed bandit game with Thompson Sampling
    and social learning from other teams.
    """

    def __init__(self, num_arms: int = 4, max_reward: int = 10,
                 social_weight: float = 0.3, num_rounds: int = 25):
        self.num_arms = num_arms
        self.max_reward = max_reward
        self.social_weight = social_weight
        self.num_rounds = num_rounds

        # Initialize arms
        self.arms = [DirichletArm(max_reward) for _ in range(num_arms)]

        # Track history
        self.round_history: List[Tuple[int, int, int]] = []  # (arm, reward, forbidden)
        self.cumulative_reward = 0

        # Social learning: track how often each arm is chosen by other teams
        self.team_choice_counts = np.zeros(num_arms)

    def record_team_choices(self, choices: List[Optional[int]]):
        """
        Record other teams' arm choices from the previous round.

        Args:
            choices: List of arm indices chosen by other teams.
                     Use None for unknown choices.
        """
        for choice in choices:
            if choice is not None and 0 <= choice < self.num_arms:
                self.team_choice_counts[choice] += 1

    def get_social_boost(self, arm_idx: int) -> np.ndarray:
        """
        Compute the social learning boost for an arm's Dirichlet prior.

        The boost is distributed across reward categories proportionally
        to the current belief (so it reinforces existing patterns).
        """
        social_count = self.team_choice_counts[arm_idx] * self.social_weight

        if social_count == 0:
            return np.zeros(self.max_reward + 1)

        # Distribute boost proportionally to current alpha
        # This means social evidence reinforces current beliefs
        current_alpha = self.arms[arm_idx].alpha
        boost = social_count * (current_alpha / np.sum(current_alpha))

        return boost

    def recommend(self, forbidden_arm: int) -> Tuple[int, List[Tuple[int, float]]]:
        """
        Run Thompson Sampling to recommend an arm.

        Args:
            forbidden_arm: Index of the arm that cannot be chosen this round

        Returns:
            Tuple of (recommended_arm, list of (arm, expected_value) for all arms)
        """
        expected_values = []

        for i in range(self.num_arms):
            if i == forbidden_arm:
                expected_values.append((i, None))  # Forbidden
            else:
                social_boost = self.get_social_boost(i)
                ev = self.arms[i].sample_expected_value(social_boost)
                expected_values.append((i, ev))

        # Find best allowed arm
        best_arm = None
        best_ev = -1
        for arm, ev in expected_values:
            if ev is not None and ev > best_ev:
                best_ev = ev
                best_arm = arm

        return best_arm, expected_values

    def record_result(self, arm: int, reward: int, forbidden_arm: int):
        """Record the result of a round."""
        self.arms[arm].update(reward)
        self.round_history.append((arm, reward, forbidden_arm))
        self.cumulative_reward += reward

    def get_arm_stats(self) -> List[dict]:
        """Get statistics for each arm."""
        stats = []
        for i, arm in enumerate(self.arms):
            stats.append({
                'arm': i,
                'pulls': arm.pull_count,
                'mean_estimate': arm.mean_estimate(),
                'social_count': int(self.team_choice_counts[i])
            })
        return stats


def parse_team_choices(input_str: str, num_arms: int) -> List[Optional[int]]:
    """
    Parse team choices input string.

    Args:
        input_str: Comma-separated values like "1,2,0,3" or "1,x,0,3"
        num_arms: Number of arms (for validation)

    Returns:
        List of arm indices (None for unknown 'x')
    """
    choices = []
    parts = input_str.strip().split(',')

    for part in parts:
        part = part.strip().lower()
        if part == 'x' or part == '':
            choices.append(None)
        else:
            try:
                arm = int(part)
                if 0 <= arm < num_arms:
                    choices.append(arm)
                else:
                    print(f"  Warning: Arm {arm} out of range, ignoring")
                    choices.append(None)
            except ValueError:
                print(f"  Warning: Could not parse '{part}', ignoring")
                choices.append(None)

    return choices


class UndoException(Exception):
    """Raised when user wants to undo (go back a step)."""
    pass


def get_valid_input(prompt: str, valid_range: range, allow_empty: bool = False,
                    allow_undo: bool = True) -> Optional[int]:
    """Get validated integer input from user. Type 'z' to undo."""
    while True:
        try:
            user_input = input(prompt).strip().lower()

            # Check for undo
            if allow_undo and user_input == 'z':
                raise UndoException()

            if allow_empty and user_input == '':
                return None

            value = int(user_input)
            if value in valid_range:
                return value
            else:
                print(f"  Please enter a number between {valid_range.start} and {valid_range.stop - 1}")
        except UndoException:
            raise  # Re-raise to be caught by caller
        except ValueError:
            if user_input == 'z':
                raise UndoException()
            print("  Please enter a valid integer (or 'z' to undo)")
        except EOFError:
            print("\n  Input ended. Exiting.")
            exit(0)


def get_team_choices_input(prompt: str, num_arms: int, allow_undo: bool = True) -> Optional[str]:
    """Get team choices input, allowing undo with 'z'."""
    try:
        user_input = input(prompt).strip()
        if allow_undo and user_input.lower() == 'z':
            raise UndoException()
        return user_input
    except EOFError:
        print("\n  Input ended. Exiting.")
        exit(0)


def print_recommendation(recommended: int, expected_values: List[Tuple[int, float]], forbidden: int):
    """Print the recommendation with expected values for all arms."""
    print(f"\n>>> RECOMMENDATION: Pull ARM {recommended}")

    # Show all arms
    arm_strs = []
    for arm, ev in expected_values:
        if arm == forbidden:
            arm_strs.append(f"Arm {arm}: FORBIDDEN")
        else:
            arm_strs.append(f"Arm {arm}: E[V]={ev:.2f}")

    print("    " + "  |  ".join(arm_strs))


def print_beliefs(game: BanditGame):
    """Print current beliefs about each arm."""
    print("\nCurrent beliefs:")
    stats = game.get_arm_stats()
    for s in stats:
        print(f"  Arm {s['arm']}: E[reward]={s['mean_estimate']:.2f} "
              f"(pulls={s['pulls']}, social={s['social_count']})")


def print_game_summary(game: BanditGame):
    """Print final game summary."""
    print("\n" + "=" * 50)
    print("GAME SUMMARY")
    print("=" * 50)

    print(f"\nTotal Reward: {game.cumulative_reward}")
    print(f"Average Reward per Round: {game.cumulative_reward / len(game.round_history):.2f}")

    print("\nPer-Arm Statistics:")
    stats = game.get_arm_stats()
    for s in stats:
        print(f"  Arm {s['arm']}: "
              f"Pulled {s['pulls']} times, "
              f"Final E[reward]={s['mean_estimate']:.2f}, "
              f"Social evidence={s['social_count']}")

    print("\nRound-by-Round History:")
    for i, (arm, reward, forbidden) in enumerate(game.round_history, 1):
        print(f"  Round {i:2d}: Pulled Arm {arm}, Got {reward:2d} (Arm {forbidden} was forbidden)")

    print("\n" + "=" * 50)
    print("Good luck with your final score!")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Thompson Sampling Bandit Strategy')
    parser.add_argument('--social-weight', type=float, default=0.3,
                        help='Weight for social learning (default: 0.3)')
    parser.add_argument('--num-arms', type=int, default=4,
                        help='Number of arms (default: 4)')
    parser.add_argument('--num-rounds', type=int, default=25,
                        help='Number of rounds (default: 25)')
    parser.add_argument('--max-reward', type=int, default=10,
                        help='Maximum reward value (default: 10)')
    args = parser.parse_args()

    # Initialize game
    game = BanditGame(
        num_arms=args.num_arms,
        max_reward=args.max_reward,
        social_weight=args.social_weight,
        num_rounds=args.num_rounds
    )

    # Print welcome message
    print("=" * 50)
    print("THOMPSON SAMPLING BANDIT STRATEGY")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Arms: {args.num_arms} (numbered 0-{args.num_arms - 1})")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Rewards: 0-{args.max_reward}")
    print(f"  Social weight: {args.social_weight}")
    print("\nInstructions:")
    print("  - Enter the forbidden arm index each round")
    print("  - From Round 2+, enter other teams' choices from previous round")
    print("  - Use 'x' for unknown team choices (e.g., 1,x,0,3)")
    print("  - Type 'z' at any prompt to UNDO and go back one step")
    print("  - Follow the recommendation to maximize your reward!")
    print("\n" + "=" * 50)

    # Main game loop
    for round_num in range(1, args.num_rounds + 1):
        # Step-based input with undo support
        # Steps: 0=team_choices (round 2+), 1=forbidden, 2=arm_pulled, 3=reward

        step = 0 if round_num > 1 else 1  # Skip team choices on round 1

        # Temporary storage for this round (not committed until complete)
        team_input = None
        forbidden = None
        recommended = None
        expected_values = None
        actual_arm = None
        reward = None

        print(f"\n{'=' * 50}")
        print(f"ROUND {round_num}/{args.num_rounds}")
        print("=" * 50)

        while True:
            try:
                # STEP 0: Team choices (Round 2+ only)
                if step == 0:
                    print(f"\n[Recording Round {round_num - 1} results from other teams]")
                    team_input = get_team_choices_input(
                        f"Other teams' choices from Round {round_num - 1} "
                        f"(e.g., 1,0,1,3 or x for unknown): ",
                        args.num_arms
                    )
                    if team_input:
                        choices = parse_team_choices(team_input, args.num_arms)
                        # Don't record yet - just validate and store
                        print(f"  Got {len([c for c in choices if c is not None])} team choices")
                    step = 1
                    continue

                # STEP 1: Forbidden arm
                if step == 1:
                    forbidden = get_valid_input(
                        f"\nForbidden arm this round (0-{args.num_arms - 1}): ",
                        range(args.num_arms)
                    )
                    # Get recommendation (regenerate each time in case we undo)
                    recommended, expected_values = game.recommend(forbidden)
                    print_recommendation(recommended, expected_values, forbidden)
                    step = 2
                    continue

                # STEP 2: Arm pulled
                if step == 2:
                    actual_arm = get_valid_input(
                        f"\nWhat arm did you pull? (0-{args.num_arms - 1}): ",
                        range(args.num_arms)
                    )
                    if actual_arm == forbidden:
                        print(f"  Warning: Arm {actual_arm} was forbidden! Recording anyway...")
                    step = 3
                    continue

                # STEP 3: Reward
                if step == 3:
                    reward = get_valid_input(
                        f"Your reward (0-{args.max_reward}): ",
                        range(args.max_reward + 1)
                    )
                    # All inputs collected - break out to commit
                    break

            except UndoException:
                # Go back one step
                if step == 0:
                    print("  (Already at first step, cannot undo further)")
                elif step == 1:
                    if round_num > 1:
                        step = 0
                        print("  << Going back to team choices")
                    else:
                        print("  (Already at first step, cannot undo further)")
                elif step == 2:
                    step = 1
                    print("  << Going back to forbidden arm")
                elif step == 3:
                    step = 2
                    print("  << Going back to arm selection")
                continue

        # NOW commit everything for this round
        if round_num > 1 and team_input:
            choices = parse_team_choices(team_input, args.num_arms)
            game.record_team_choices(choices)

        game.record_result(actual_arm, reward, forbidden)

        print(f"\n[check] Round {round_num} complete. Cumulative reward: {game.cumulative_reward}")

        # Show updated beliefs
        print_beliefs(game)

    # Print final summary
    print_game_summary(game)


if __name__ == "__main__":
    main()
