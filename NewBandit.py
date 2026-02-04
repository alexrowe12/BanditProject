#!/usr/bin/env python3
"""
Information-Directed Sampling (IDS) for Multi-Armed Bandit
An optimal strategy for finite-horizon categorical bandits with social learning.

This is a more sophisticated alternative to Thompson Sampling that explicitly
optimizes the exploration-exploitation tradeoff by minimizing the information ratio.

Usage: python NewBandit.py [--num-rounds 25] [--verbose]
"""

import numpy as np
import argparse
from typing import List, Optional, Tuple


# =============================================================================
# MATH UTILITIES (replacing scipy dependency)
# =============================================================================

def gammaln(x):
    """Log of gamma function using Stirling's approximation for arrays."""
    x = np.asarray(x, dtype=float)
    # Use numpy's built-in for scalar or small arrays
    # Stirling approximation: ln(Gamma(x)) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2*pi)
    # More accurate version with correction terms
    return (
        (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi)
        + 1/(12*x) - 1/(360*x**3)
    )


def digamma(x):
    """Digamma function (psi) - derivative of log gamma."""
    x = np.asarray(x, dtype=float)
    # Approximation: psi(x) ≈ ln(x) - 1/(2x) - 1/(12x^2) for large x
    # For better accuracy, use recurrence: psi(x+1) = psi(x) + 1/x
    result = np.zeros_like(x)

    # Shift small values up using recurrence
    mask = x < 6
    shift = np.zeros_like(x)
    x_shifted = x.copy()

    while np.any(x_shifted < 6):
        small = x_shifted < 6
        shift[small] -= 1.0 / x_shifted[small]
        x_shifted[small] += 1

    # Asymptotic expansion for x >= 6
    result = (
        np.log(x_shifted) - 1/(2*x_shifted) - 1/(12*x_shifted**2)
        + 1/(120*x_shifted**4) - 1/(252*x_shifted**6)
    )

    return result + shift


# =============================================================================
# CORE CLASSES
# =============================================================================

class CategoricalArm:
    """
    Represents a single arm with a Dirichlet prior over categorical rewards.
    Tracks beliefs about reward distribution and provides information-theoretic metrics.
    """

    def __init__(self, max_reward: int = 10):
        self.max_reward = max_reward
        self.num_categories = max_reward + 1
        # Uniform Dirichlet prior
        self.alpha = np.ones(self.num_categories)
        self.pull_count = 0

    def update(self, reward: int):
        """Update posterior after observing a reward."""
        if 0 <= reward <= self.max_reward:
            self.alpha[reward] += 1
            self.pull_count += 1
        else:
            raise ValueError(f"Reward must be between 0 and {self.max_reward}")

    def expected_value(self) -> float:
        """Compute expected reward under current posterior mean."""
        # Posterior mean of Dirichlet is alpha / sum(alpha)
        probs = self.alpha / np.sum(self.alpha)
        rewards = np.arange(self.num_categories)
        return np.sum(probs * rewards)

    def sample_expected_value(self) -> float:
        """Thompson-style: sample distribution, compute expected value."""
        sampled_probs = np.random.dirichlet(self.alpha)
        rewards = np.arange(self.num_categories)
        return np.sum(sampled_probs * rewards)

    def variance(self) -> float:
        """Variance of the expected value estimate (uncertainty)."""
        alpha_sum = np.sum(self.alpha)
        probs = self.alpha / alpha_sum
        rewards = np.arange(self.num_categories)

        # E[X^2] - E[X]^2
        ex = np.sum(probs * rewards)
        ex2 = np.sum(probs * rewards**2)
        return ex2 - ex**2

    def dirichlet_entropy(self) -> float:
        """Entropy of the Dirichlet distribution (uncertainty in belief)."""
        alpha = self.alpha
        alpha_sum = np.sum(alpha)
        k = len(alpha)

        # Correct Dirichlet entropy formula:
        # H = ln(B(α)) + (α₀ - K)·ψ(α₀) - Σ(αᵢ - 1)·ψ(αᵢ)
        # where ln(B(α)) = Σ gammaln(αᵢ) - gammaln(α₀)
        entropy = (
            np.sum(gammaln(alpha)) - gammaln(alpha_sum)
            + (alpha_sum - k) * digamma(alpha_sum)
            - np.sum((alpha - 1) * digamma(alpha))
        )
        return entropy

    def information_gain(self) -> float:
        """
        Expected information gain from pulling this arm.
        Measures expected reduction in entropy of our belief.
        """
        alpha = self.alpha
        alpha_sum = np.sum(alpha)
        k = self.num_categories

        # Current entropy
        current_entropy = self.dirichlet_entropy()

        # Expected entropy after one observation
        # For each possible reward, compute new entropy and weight by probability
        expected_entropy = 0.0
        for reward in range(k):
            # Probability of this reward under current belief
            p_reward = alpha[reward] / alpha_sum

            # New alpha after observing this reward
            new_alpha = alpha.copy()
            new_alpha[reward] += 1
            new_alpha_sum = alpha_sum + 1

            # Entropy of updated Dirichlet (correct formula)
            new_entropy = (
                np.sum(gammaln(new_alpha)) - gammaln(new_alpha_sum)
                + (new_alpha_sum - k) * digamma(new_alpha_sum)
                - np.sum((new_alpha - 1) * digamma(new_alpha))
            )

            expected_entropy += p_reward * new_entropy

        # Information gain = reduction in entropy
        return current_entropy - expected_entropy


class SocialModel:
    """
    Models other teams' behavior to extract additional signal.
    Uses decay-weighted evidence from observed arm choices.
    """

    def __init__(self, num_teams: int = 4, num_arms: int = 4, decay: float = 0.9):
        self.num_teams = num_teams
        self.num_arms = num_arms
        self.decay = decay
        self.history: List[List[Optional[int]]] = []
        self.choice_counts = np.zeros(num_arms)

    def update(self, team_choices: List[Optional[int]]):
        """Record team choices from previous round."""
        self.history.append(team_choices)
        for choice in team_choices:
            if choice is not None and 0 <= choice < self.num_arms:
                self.choice_counts[choice] += 1

    def get_social_evidence(self, arm: int) -> float:
        """
        Compute decay-weighted social evidence for an arm.
        Recent choices weighted higher than older ones.
        """
        evidence = 0.0
        weight = 1.0

        for round_choices in reversed(self.history):
            for choice in round_choices:
                if choice == arm:
                    evidence += weight
            weight *= self.decay

        return evidence

    def get_social_boost(self, arm: int, base_weight: float = 0.2) -> float:
        """Convert social evidence to a boost for the arm's expected value."""
        evidence = self.get_social_evidence(arm)
        # Normalize by total evidence to get relative preference
        total_evidence = sum(self.get_social_evidence(a) for a in range(self.num_arms))
        if total_evidence == 0:
            return 0.0
        return base_weight * (evidence / total_evidence) * self.num_arms

    def get_consensus_arm(self) -> Optional[int]:
        """Identify if teams are converging on a particular arm."""
        if len(self.history) < 3:
            return None

        # Look at last 3 rounds
        recent_counts = np.zeros(self.num_arms)
        for round_choices in self.history[-3:]:
            for choice in round_choices:
                if choice is not None:
                    recent_counts[choice] += 1

        max_count = np.max(recent_counts)
        # Consensus if one arm has > 50% of choices
        if max_count > 0.5 * np.sum(recent_counts):
            return int(np.argmax(recent_counts))
        return None


class OptimalBanditGame:
    """
    Information-Directed Sampling bandit with horizon-awareness and social learning.
    """

    def __init__(self, num_arms: int = 4, max_reward: int = 10,
                 num_rounds: int = 25, social_weight: float = 0.2,
                 social_decay: float = 0.9, verbose: bool = False):
        self.num_arms = num_arms
        self.max_reward = max_reward
        self.num_rounds = num_rounds
        self.social_weight = social_weight
        self.verbose = verbose

        # Initialize arms
        self.arms = [CategoricalArm(max_reward) for _ in range(num_arms)]

        # Social model
        self.social_model = SocialModel(num_teams=4, num_arms=num_arms, decay=social_decay)

        # Game state
        self.current_round = 0
        self.round_history: List[Tuple[int, int, int]] = []
        self.cumulative_reward = 0

    def get_temperature(self) -> float:
        """
        Horizon-aware temperature for exploration/exploitation balance.
        High early (explore), low late (exploit).
        """
        if self.current_round == 0:
            return 1.0

        remaining = self.num_rounds - self.current_round
        # Linear decay from 1.0 to 0.1
        temp = max(0.1, remaining / self.num_rounds)
        return temp

    def compute_information_ratio(self, arm: int, forbidden: int) -> Tuple[float, float, float]:
        """
        Compute the information ratio for an arm.
        Returns (ratio, regret, info_gain) for analysis.
        """
        # Expected value of this arm (with social boost)
        ev = self.arms[arm].expected_value()
        social_boost = self.social_model.get_social_boost(arm, self.social_weight)
        adjusted_ev = ev + social_boost

        # Find best expected value among allowed arms
        best_ev = -float('inf')
        for a in range(self.num_arms):
            if a == forbidden:
                continue
            a_ev = self.arms[a].expected_value()
            a_boost = self.social_model.get_social_boost(a, self.social_weight)
            if a_ev + a_boost > best_ev:
                best_ev = a_ev + a_boost

        # Expected regret (how much worse than best)
        regret = max(0, best_ev - adjusted_ev)

        # Information gain from pulling this arm
        info_gain = self.arms[arm].information_gain()

        # Safety check: info_gain should be positive
        # (numerical issues with digamma approximation can cause problems)
        info_gain = max(info_gain, 1e-6)

        # Information ratio: regret^2 / info_gain
        # Lower is better (low regret per unit of information)
        epsilon = 1e-10
        ratio = (regret ** 2) / (info_gain + epsilon)

        return ratio, regret, info_gain

    def recommend_ids(self, forbidden: int) -> Tuple[int, List[dict]]:
        """
        Information-Directed Sampling: select arm minimizing information ratio.

        Returns:
            (recommended_arm, details for each arm)
        """
        temperature = self.get_temperature()
        arm_details = []

        best_arms = []  # Track all arms with best score for tie-breaking
        best_score = float('inf')

        for arm in range(self.num_arms):
            if arm == forbidden:
                arm_details.append({
                    'arm': arm,
                    'forbidden': True,
                    'ev': None,
                    'info_gain': None,
                    'regret': None,
                    'ratio': None,
                    'score': None
                })
                continue

            ratio, regret, info_gain = self.compute_information_ratio(arm, forbidden)
            ev = self.arms[arm].expected_value()
            social_boost = self.social_model.get_social_boost(arm, self.social_weight)
            adjusted_ev = ev + social_boost

            # Temperature-adjusted score
            # Key insight: when regret is near 0, ratio is ~0 for all arms
            # In that case, prefer arms with HIGH information gain (exploration)
            #
            # Score formula: ratio - (exploration_weight * info_gain)
            # - Lower score is better
            # - Low ratio (low regret per info) = good
            # - High info_gain with exploration bonus = lower score = good

            exploration_weight = temperature * 0.5  # Decays as game progresses

            if regret < 0.01:
                # Near-zero regret: pure exploration mode
                # Prefer arms we know less about (higher info_gain)
                score = -info_gain * exploration_weight
            else:
                # Normal IDS: balance regret vs information
                score = ratio - (exploration_weight * info_gain)

            arm_details.append({
                'arm': arm,
                'forbidden': False,
                'ev': adjusted_ev,
                'info_gain': info_gain,
                'regret': regret,
                'ratio': ratio,
                'score': score
            })

            if score < best_score - 1e-9:  # Strictly better
                best_score = score
                best_arms = [arm]
            elif abs(score - best_score) < 1e-9:  # Tie
                best_arms.append(arm)

        # Random tie-breaking
        best_arm = np.random.choice(best_arms) if best_arms else 0

        return best_arm, arm_details

    def record_team_choices(self, choices: List[Optional[int]]):
        """Record other teams' choices from previous round."""
        self.social_model.update(choices)

    def record_result(self, arm: int, reward: int, forbidden: int):
        """Record result of a round."""
        self.arms[arm].update(reward)
        self.round_history.append((arm, reward, forbidden))
        self.cumulative_reward += reward
        self.current_round += 1

    def get_arm_stats(self) -> List[dict]:
        """Get statistics for each arm."""
        stats = []
        for i, arm in enumerate(self.arms):
            stats.append({
                'arm': i,
                'pulls': arm.pull_count,
                'ev': arm.expected_value(),
                'variance': arm.variance(),
                'info_gain': arm.information_gain(),
                'social': self.social_model.get_social_evidence(i)
            })
        return stats


# =============================================================================
# INPUT HANDLING (with undo support)
# =============================================================================

class UndoException(Exception):
    """Raised when user wants to undo."""
    pass


def get_valid_input(prompt: str, valid_range: range, allow_undo: bool = True) -> int:
    """Get validated integer input. Type 'z' to undo."""
    while True:
        try:
            user_input = input(prompt).strip().lower()

            if allow_undo and user_input == 'z':
                raise UndoException()

            value = int(user_input)
            if value in valid_range:
                return value
            else:
                print(f"  Please enter a number between {valid_range.start} and {valid_range.stop - 1}")
        except UndoException:
            raise
        except ValueError:
            if user_input == 'z':
                raise UndoException()
            print("  Please enter a valid integer (or 'z' to undo)")
        except EOFError:
            print("\n  Exiting.")
            exit(0)


def get_team_choices_input(prompt: str, allow_undo: bool = True) -> str:
    """Get team choices input string."""
    try:
        user_input = input(prompt).strip()
        if allow_undo and user_input.lower() == 'z':
            raise UndoException()
        return user_input
    except EOFError:
        print("\n  Exiting.")
        exit(0)


def parse_team_choices(input_str: str, num_arms: int) -> List[Optional[int]]:
    """Parse comma-separated team choices (1-4). Use 'x' for unknown."""
    choices = []
    for part in input_str.split(','):
        part = part.strip().lower()
        if part == 'x' or part == '':
            choices.append(None)
        else:
            try:
                arm = int(part) - 1  # Convert 1-4 to 0-3
                if 0 <= arm < num_arms:
                    choices.append(arm)
                else:
                    choices.append(None)
            except ValueError:
                choices.append(None)
    return choices


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_recommendation(recommended: int, arm_details: List[dict], forbidden: int, verbose: bool):
    """Print the IDS recommendation."""
    print(f"\n>>> RECOMMENDATION: Pull ARM {recommended + 1}")  # Display as 1-4

    # Compact view
    parts = []
    for d in arm_details:
        if d['forbidden']:
            parts.append(f"Arm {d['arm'] + 1}: FORBIDDEN")
        else:
            parts.append(f"Arm {d['arm'] + 1}: E[V]={d['ev']:.2f}")
    print("    " + "  |  ".join(parts))

    # Verbose: show IDS details
    if verbose:
        print("\n    [IDS Analysis]")
        print(f"    {'Arm':<5} {'E[V]':<8} {'InfoGain':<10} {'Regret':<8} {'Ratio':<10} {'Score':<10}")
        print("    " + "-" * 51)
        for d in arm_details:
            if d['forbidden']:
                print(f"    {d['arm'] + 1:<5} {'--':<8} {'--':<10} {'--':<8} {'--':<10} FORBIDDEN")
            else:
                print(f"    {d['arm'] + 1:<5} {d['ev']:<8.3f} {d['info_gain']:<10.4f} "
                      f"{d['regret']:<8.3f} {d['ratio']:<10.4f} {d['score']:<10.4f}")


def print_beliefs(game: OptimalBanditGame):
    """Print current beliefs."""
    print("\nCurrent beliefs:")
    stats = game.get_arm_stats()
    for s in stats:
        print(f"  Arm {s['arm'] + 1}: E[V]={s['ev']:.2f} "
              f"(pulls={s['pulls']}, var={s['variance']:.2f}, social={s['social']:.1f})")


def print_game_summary(game: OptimalBanditGame):
    """Print final summary."""
    print("\n" + "=" * 55)
    print("GAME SUMMARY - Information-Directed Sampling")
    print("=" * 55)

    print(f"\nTotal Reward: {game.cumulative_reward}")
    print(f"Average per Round: {game.cumulative_reward / game.num_rounds:.2f}")

    print("\nFinal Arm Statistics:")
    stats = game.get_arm_stats()
    for s in stats:
        print(f"  Arm {s['arm'] + 1}: Pulled {s['pulls']}x, "
              f"Final E[V]={s['ev']:.2f}, Social={s['social']:.1f}")

    # Consensus detection
    consensus = game.social_model.get_consensus_arm()
    if consensus is not None:
        print(f"\nOther teams converged on: Arm {consensus + 1}")

    print("\nRound History:")
    for i, (arm, reward, forbidden) in enumerate(game.round_history, 1):
        print(f"  Round {i:2d}: Pulled Arm {arm + 1}, Got {reward:2d} (Arm {forbidden + 1} forbidden)")

    print("\n" + "=" * 55)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Information-Directed Sampling Bandit Strategy (Optimal)')
    parser.add_argument('--num-arms', type=int, default=4)
    parser.add_argument('--num-rounds', type=int, default=25)
    parser.add_argument('--max-reward', type=int, default=10)
    parser.add_argument('--social-weight', type=float, default=0.2)
    parser.add_argument('--social-decay', type=float, default=0.9)
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed IDS analysis each round')
    args = parser.parse_args()

    game = OptimalBanditGame(
        num_arms=args.num_arms,
        max_reward=args.max_reward,
        num_rounds=args.num_rounds,
        social_weight=args.social_weight,
        social_decay=args.social_decay,
        verbose=args.verbose
    )

    # Welcome
    print("=" * 55)
    print("INFORMATION-DIRECTED SAMPLING BANDIT STRATEGY")
    print("=" * 55)
    print(f"\nConfiguration:")
    print(f"  Arms: {args.num_arms} (numbered 1-{args.num_arms})")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Rewards: 0-{args.max_reward}")
    print(f"  Social weight: {args.social_weight}, decay: {args.social_decay}")
    print(f"  Verbose: {args.verbose}")
    print("\nInstructions:")
    print("  - Type 'z' at any prompt to UNDO and go back one step")
    print("  - From Round 2+, enter other teams' previous choices")
    print("  - Use 'x' for unknown team choices (e.g., 1,x,2,3)")
    print("\n" + "=" * 55)

    # Main loop
    for round_num in range(1, args.num_rounds + 1):
        step = 0 if round_num > 1 else 1

        team_input = None
        forbidden = None
        actual_arm = None
        reward = None

        print(f"\n{'=' * 55}")
        print(f"ROUND {round_num}/{args.num_rounds}  |  Temperature: {game.get_temperature():.2f}")
        print("=" * 55)

        while True:
            try:
                # Step 0: Team choices (Round 2+)
                if step == 0:
                    print(f"\n[Recording Round {round_num - 1} results from other teams]")
                    team_input = get_team_choices_input(
                        f"Other teams' choices from Round {round_num - 1} "
                        f"(e.g., 1,2,1,3 or x): "
                    )
                    if team_input:
                        choices = parse_team_choices(team_input, args.num_arms)
                        print(f"  Got {len([c for c in choices if c is not None])} team choices")
                    step = 1
                    continue

                # Step 1: Forbidden arm
                if step == 1:
                    forbidden = get_valid_input(
                        f"\nForbidden arm this round (1-{args.num_arms}): ",
                        range(1, args.num_arms + 1)
                    ) - 1  # Convert 1-4 to 0-3

                    # Get IDS recommendation
                    recommended, arm_details = game.recommend_ids(forbidden)
                    print_recommendation(recommended, arm_details, forbidden, args.verbose)
                    step = 2
                    continue

                # Step 2: Arm pulled
                if step == 2:
                    actual_arm = get_valid_input(
                        f"\nWhat arm did you pull? (1-{args.num_arms}): ",
                        range(1, args.num_arms + 1)
                    ) - 1  # Convert 1-4 to 0-3
                    if actual_arm == forbidden:
                        print(f"  Warning: Arm {actual_arm + 1} was forbidden!")
                    step = 3
                    continue

                # Step 3: Reward
                if step == 3:
                    reward = get_valid_input(
                        f"Your reward (0-{args.max_reward}): ",
                        range(args.max_reward + 1)
                    )
                    break

            except UndoException:
                if step == 0:
                    print("  (Already at first step)")
                elif step == 1:
                    if round_num > 1:
                        step = 0
                        print("  << Back to team choices")
                    else:
                        print("  (Already at first step)")
                elif step == 2:
                    step = 1
                    print("  << Back to forbidden arm")
                elif step == 3:
                    step = 2
                    print("  << Back to arm selection")
                continue

        # Commit round
        if round_num > 1 and team_input:
            choices = parse_team_choices(team_input, args.num_arms)
            game.record_team_choices(choices)

        game.record_result(actual_arm, reward, forbidden)

        print(f"\n[OK] Round {round_num} complete. Cumulative: {game.cumulative_reward}")
        print_beliefs(game)

    print_game_summary(game)


if __name__ == "__main__":
    main()
