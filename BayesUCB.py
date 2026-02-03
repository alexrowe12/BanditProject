#!/usr/bin/env python3
"""
Bayes-UCB with Top-2 Objective for Multi-Armed Bandit

A strategy optimized for the 25-round game where one arm is randomly forbidden
each round. Key features:
- Gaussian posterior model with Welford's online variance
- UCB scoring with quantile schedule (more exploit over time)
- Top-2 objective: explicitly optimizes for having good #1 AND #2 arms
- Warm-start: forces exploration of unpulled arms first

Usage: python BayesUCB.py [--c 2.0] [--beta 0.5] [--var-floor 1.0] [--verbose]
"""

import numpy as np
import argparse
import math
from typing import List, Optional, Tuple


# =============================================================================
# MATH UTILITIES (no scipy)
# =============================================================================

def norm_ppf(q: float) -> float:
    """
    Inverse of standard normal CDF (quantile function).
    Returns z such that P(Z < z) = q.

    Uses rational approximation from Abramowitz and Stegun (1964).
    Accurate to about 4.5e-4.
    """
    if q <= 0:
        return -float('inf')
    if q >= 1:
        return float('inf')
    if q == 0.5:
        return 0.0

    # Use symmetry for q < 0.5
    if q < 0.5:
        return -norm_ppf(1 - q)

    # Rational approximation for 0.5 < q < 1
    # First transform to t where 0 < t < 0.5
    t = 1 - q

    # Coefficients for rational approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    # Compute approximation
    w = math.sqrt(-2 * math.log(t))
    z = w - (c0 + c1*w + c2*w*w) / (1 + d1*w + d2*w*w + d3*w*w*w)

    return z


# =============================================================================
# CORE CLASSES
# =============================================================================

class GaussianArm:
    """
    Represents a single arm with Gaussian posterior on mean reward.
    Uses Welford's algorithm for online mean/variance calculation.
    """

    def __init__(self, prior_mean: float = 5.0, prior_var: float = 10.0):
        self.prior_mean = prior_mean
        self.prior_var = prior_var

        # Welford's algorithm state
        self.n = 0            # Number of pulls
        self.mean = 0.0       # Running sample mean
        self.M2 = 0.0         # Sum of squared deviations
        self.variance = 0.0   # Sample variance

    def update(self, reward: int):
        """Update running mean and variance using Welford's algorithm."""
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        delta2 = reward - self.mean
        self.M2 += delta * delta2

        if self.n > 1:
            self.variance = self.M2 / (self.n - 1)
        else:
            self.variance = 0.0  # Will use var_floor in posterior

    def get_posterior_params(self, var_floor: float = 1.0) -> Tuple[float, float]:
        """
        Get posterior mean and standard deviation.

        For simplicity (and because prior is vague), we approximate:
        - m ≈ sample mean (or prior mean if no observations)
        - sd ≈ sqrt(max(s², var_floor) / n) for standard error with floor

        Returns:
            (posterior_mean, posterior_sd)
        """
        if self.n == 0:
            # No observations: use prior
            return self.prior_mean, math.sqrt(self.prior_var)

        # Posterior mean is approximately sample mean (with vague prior)
        m = self.mean

        # Posterior SD: standard error with variance floor
        sigma_sq = max(self.variance, var_floor)
        sd = math.sqrt(sigma_sq / self.n)

        return m, sd

    def sample_posterior(self, n_samples: int, var_floor: float = 1.0) -> np.ndarray:
        """
        Sample from the posterior distribution of the mean.

        Returns:
            Array of n_samples posterior samples
        """
        m, sd = self.get_posterior_params(var_floor)
        return np.random.normal(m, sd, n_samples)

    def expected_value(self, var_floor: float = 1.0) -> float:
        """Get posterior expected value (mean)."""
        m, _ = self.get_posterior_params(var_floor)
        return m


class BayesUCBGame:
    """
    Bayes-UCB strategy with Top-2 objective for forbidden-arm bandit.

    Key features:
    - UCB scoring: selects optimistic arm using posterior quantiles
    - Top-2 objective: bonus for arms likely to be in top-2 (handles forbidden arm)
    - Warm-start: forces exploration of unpulled arms
    - Quantile schedule: q(t) = 1 - 1/(t+1)^c increases over time (exploit late)
    """

    def __init__(self, num_arms: int = 4, max_reward: int = 10, num_rounds: int = 25,
                 c: float = 2.0, beta: float = 0.5, var_floor: float = 1.0,
                 prior_mean: float = 5.0, prior_var: float = 10.0,
                 verbose: bool = False):
        self.num_arms = num_arms
        self.max_reward = max_reward
        self.num_rounds = num_rounds

        # Hyperparameters
        self.c = c                  # Quantile schedule exponent
        self.beta = beta            # Top-2 probability weight
        self.var_floor = var_floor  # Minimum variance (prevents overconfidence)
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.verbose = verbose

        # Initialize arms
        self.arms = [GaussianArm(prior_mean, prior_var) for _ in range(num_arms)]

        # Game state
        self.current_round = 0
        self.round_history: List[Tuple[int, int, int]] = []  # (arm, reward, forbidden)
        self.cumulative_reward = 0

    def get_quantile(self, t: int) -> float:
        """
        Quantile schedule: q(t) = 1 - 1/(t+1)^c

        Higher quantile = more exploitation
        c controls how fast we transition to exploitation
        """
        return 1 - 1 / ((t + 1) ** self.c)

    def compute_ucb(self, arm_idx: int, t: int) -> float:
        """
        Compute UCB score for an arm.
        UCB_i(t) = m_i + z_q(t) × sd_i
        """
        m, sd = self.arms[arm_idx].get_posterior_params(self.var_floor)
        q = self.get_quantile(t)
        z_q = norm_ppf(q)
        return m + z_q * sd

    def compute_top2_probs(self, n_samples: int = 100) -> np.ndarray:
        """
        Estimate P(arm is in top-2) via Monte Carlo.

        Sample from each arm's posterior, rank, count how often each arm is #1 or #2.
        """
        counts = np.zeros(self.num_arms)

        for _ in range(n_samples):
            # Sample one mean from each arm's posterior
            samples = [arm.sample_posterior(1, self.var_floor)[0] for arm in self.arms]

            # Find top-2 indices
            top2 = np.argsort(samples)[-2:]  # Indices of 2 largest
            counts[top2] += 1

        return counts / n_samples  # Normalize to probabilities

    def recommend(self, forbidden: int) -> Tuple[int, List[dict]]:
        """
        Main decision logic: Bayes-UCB with Top-2 objective.

        Returns:
            (recommended_arm, details_for_each_arm)
        """
        t = self.current_round
        arm_details = []

        # Forced warm-start: pull unpulled arms first
        for arm in range(self.num_arms):
            if arm != forbidden and self.arms[arm].n == 0:
                # Build details for display
                for i in range(self.num_arms):
                    if i == forbidden:
                        arm_details.append({
                            'arm': i, 'forbidden': True,
                            'm': None, 'sd': None, 'ucb': None,
                            'top2_prob': None, 'final_score': None
                        })
                    else:
                        m, sd = self.arms[i].get_posterior_params(self.var_floor)
                        arm_details.append({
                            'arm': i, 'forbidden': False,
                            'm': m, 'sd': sd, 'ucb': None,
                            'top2_prob': None, 'final_score': None,
                            'note': 'warm-start' if i == arm else None
                        })
                return arm, arm_details

        # Compute UCB scores
        ucb_scores = []
        for i in range(self.num_arms):
            if i == forbidden:
                ucb_scores.append(-float('inf'))
            else:
                ucb_scores.append(self.compute_ucb(i, t))

        # Compute Top-2 probabilities
        top2_probs = self.compute_top2_probs(n_samples=100)

        # Final scores: UCB + beta * top2_prob
        final_scores = []
        for i in range(self.num_arms):
            if i == forbidden:
                final_scores.append(-float('inf'))
            else:
                final_scores.append(ucb_scores[i] + self.beta * top2_probs[i])

        # Build arm details
        for i in range(self.num_arms):
            if i == forbidden:
                arm_details.append({
                    'arm': i, 'forbidden': True,
                    'm': None, 'sd': None, 'ucb': None,
                    'top2_prob': None, 'final_score': None
                })
            else:
                m, sd = self.arms[i].get_posterior_params(self.var_floor)
                arm_details.append({
                    'arm': i, 'forbidden': False,
                    'm': m, 'sd': sd, 'ucb': ucb_scores[i],
                    'top2_prob': top2_probs[i], 'final_score': final_scores[i]
                })

        # Select best arm
        best_arm = int(np.argmax(final_scores))

        return best_arm, arm_details

    def record_result(self, arm: int, reward: int, forbidden: int):
        """Record result of a round."""
        self.arms[arm].update(reward)
        self.round_history.append((arm, reward, forbidden))
        self.cumulative_reward += reward
        self.current_round += 1

    def record_team_choices(self, choices: List[Optional[int]]):
        """No-op: Bayes-UCB doesn't use social learning."""
        pass

    def get_arm_stats(self) -> List[dict]:
        """Get statistics for each arm (for Monte Carlo harness)."""
        stats = []
        for i, arm in enumerate(self.arms):
            m, sd = arm.get_posterior_params(self.var_floor)
            stats.append({
                'arm': i,
                'pulls': arm.n,
                'ev': m,
                'sd': sd,
                'variance': arm.variance
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
    """Get team choices input string (for interface compatibility)."""
    try:
        user_input = input(prompt).strip()
        if allow_undo and user_input.lower() == 'z':
            raise UndoException()
        return user_input
    except EOFError:
        print("\n  Exiting.")
        exit(0)


def parse_team_choices(input_str: str, num_arms: int) -> List[Optional[int]]:
    """Parse comma-separated team choices. Use 'x' for unknown."""
    choices = []
    for part in input_str.split(','):
        part = part.strip().lower()
        if part == 'x' or part == '':
            choices.append(None)
        else:
            try:
                arm = int(part)
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
    """Print the Bayes-UCB recommendation."""
    print(f"\n>>> RECOMMENDATION: Pull ARM {recommended}")

    # Compact view
    parts = []
    for d in arm_details:
        if d['forbidden']:
            parts.append(f"Arm {d['arm']}: FORBIDDEN")
        elif d.get('note') == 'warm-start':
            parts.append(f"Arm {d['arm']}: E[V]={d['m']:.2f} (warm-start)")
        else:
            parts.append(f"Arm {d['arm']}: E[V]={d['m']:.2f}")
    print("    " + "  |  ".join(parts))

    # Verbose: show Bayes-UCB details
    if verbose:
        print("\n    [Bayes-UCB Analysis]")
        print(f"    {'Arm':<5} {'Mean':<8} {'SD':<8} {'UCB':<10} {'Top2 Prob':<10} {'Final':<10}")
        print("    " + "-" * 51)
        for d in arm_details:
            if d['forbidden']:
                print(f"    {d['arm']:<5} {'--':<8} {'--':<8} {'--':<10} {'--':<10} FORBIDDEN")
            elif d['ucb'] is None:
                print(f"    {d['arm']:<5} {d['m']:<8.3f} {d['sd']:<8.3f} {'(warm)':<10} {'--':<10} {'--':<10}")
            else:
                print(f"    {d['arm']:<5} {d['m']:<8.3f} {d['sd']:<8.3f} "
                      f"{d['ucb']:<10.3f} {d['top2_prob']:<10.3f} {d['final_score']:<10.3f}")


def print_beliefs(game: BayesUCBGame):
    """Print current beliefs."""
    print("\nCurrent beliefs:")
    stats = game.get_arm_stats()
    for s in stats:
        print(f"  Arm {s['arm']}: E[V]={s['ev']:.2f} "
              f"(pulls={s['pulls']}, sd={s['sd']:.2f})")


def print_game_summary(game: BayesUCBGame):
    """Print final summary."""
    print("\n" + "=" * 55)
    print("GAME SUMMARY - Bayes-UCB with Top-2 Objective")
    print("=" * 55)

    print(f"\nHyperparameters: c={game.c}, beta={game.beta}, var_floor={game.var_floor}")
    print(f"Total Reward: {game.cumulative_reward}")
    print(f"Average per Round: {game.cumulative_reward / game.num_rounds:.2f}")

    print("\nFinal Arm Statistics:")
    stats = game.get_arm_stats()
    for s in stats:
        print(f"  Arm {s['arm']}: Pulled {s['pulls']}x, "
              f"Final E[V]={s['ev']:.2f} (sd={s['sd']:.2f})")

    print("\nRound History:")
    for i, (arm, reward, forbidden) in enumerate(game.round_history, 1):
        print(f"  Round {i:2d}: Pulled Arm {arm}, Got {reward:2d} (Arm {forbidden} forbidden)")

    print("\n" + "=" * 55)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bayes-UCB with Top-2 Objective for Multi-Armed Bandit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hyperparameters:
  --c         Quantile schedule exponent (default: 2.0)
              Higher = exploit earlier
  --beta      Top-2 probability weight (default: 0.5)
              Higher = more focus on backup arm quality
  --var-floor Minimum variance (default: 1.0)
              Prevents overconfidence with few observations

Examples:
  python BayesUCB.py                          # Default hyperparameters
  python BayesUCB.py --c 2.5 --beta 0.8       # Custom hyperparameters
  python BayesUCB.py --verbose                # Show detailed scoring
        """
    )
    parser.add_argument('--num-arms', type=int, default=4)
    parser.add_argument('--num-rounds', type=int, default=25)
    parser.add_argument('--max-reward', type=int, default=10)
    parser.add_argument('--c', type=float, default=2.0,
                        help='Quantile schedule exponent (default: 2.0)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Top-2 probability weight (default: 0.5)')
    parser.add_argument('--var-floor', type=float, default=1.0,
                        help='Minimum variance (default: 1.0)')
    parser.add_argument('--prior-mean', type=float, default=5.0,
                        help='Prior mean (default: 5.0)')
    parser.add_argument('--prior-var', type=float, default=10.0,
                        help='Prior variance (default: 10.0)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed Bayes-UCB analysis each round')
    args = parser.parse_args()

    game = BayesUCBGame(
        num_arms=args.num_arms,
        max_reward=args.max_reward,
        num_rounds=args.num_rounds,
        c=args.c,
        beta=args.beta,
        var_floor=args.var_floor,
        prior_mean=args.prior_mean,
        prior_var=args.prior_var,
        verbose=args.verbose
    )

    # Welcome
    print("=" * 55)
    print("BAYES-UCB WITH TOP-2 OBJECTIVE")
    print("=" * 55)
    print(f"\nConfiguration:")
    print(f"  Arms: {args.num_arms} (numbered 0-{args.num_arms - 1})")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Rewards: 0-{args.max_reward}")
    print(f"\nHyperparameters:")
    print(f"  c (quantile exponent): {args.c}")
    print(f"  beta (top-2 weight):   {args.beta}")
    print(f"  var_floor:             {args.var_floor}")
    print(f"  Verbose: {args.verbose}")
    print("\nInstructions:")
    print("  - Type 'z' at any prompt to UNDO and go back one step")
    print("  - From Round 2+, enter other teams' previous choices (optional)")
    print("  - Use 'x' for unknown team choices (e.g., 1,x,0,3)")
    print("\n" + "=" * 55)

    # Main loop
    for round_num in range(1, args.num_rounds + 1):
        step = 0 if round_num > 1 else 1

        team_input = None
        forbidden = None
        actual_arm = None
        reward = None

        print(f"\n{'=' * 55}")
        q = game.get_quantile(game.current_round)
        print(f"ROUND {round_num}/{args.num_rounds}  |  Quantile: {q:.3f} (z={norm_ppf(q):.2f})")
        print("=" * 55)

        while True:
            try:
                # Step 0: Team choices (Round 2+) - for interface compatibility
                if step == 0:
                    print(f"\n[Recording Round {round_num - 1} results from other teams]")
                    team_input = get_team_choices_input(
                        f"Other teams' choices from Round {round_num - 1} "
                        f"(e.g., 1,0,1,3 or x, or press Enter to skip): "
                    )
                    if team_input:
                        choices = parse_team_choices(team_input, args.num_arms)
                        print(f"  Got {len([c for c in choices if c is not None])} team choices (not used by Bayes-UCB)")
                    step = 1
                    continue

                # Step 1: Forbidden arm
                if step == 1:
                    forbidden = get_valid_input(
                        f"\nForbidden arm this round (0-{args.num_arms - 1}): ",
                        range(args.num_arms)
                    )

                    # Get recommendation
                    recommended, arm_details = game.recommend(forbidden)
                    print_recommendation(recommended, arm_details, forbidden, args.verbose)
                    step = 2
                    continue

                # Step 2: Arm pulled
                if step == 2:
                    actual_arm = get_valid_input(
                        f"\nWhat arm did you pull? (0-{args.num_arms - 1}): ",
                        range(args.num_arms)
                    )
                    if actual_arm == forbidden:
                        print(f"  Warning: Arm {actual_arm} was forbidden!")
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
        game.record_result(actual_arm, reward, forbidden)

        print(f"\n[OK] Round {round_num} complete. Cumulative: {game.cumulative_reward}")
        print_beliefs(game)

    print_game_summary(game)


if __name__ == "__main__":
    main()
