#!/usr/bin/env python3
"""
Monte Carlo A/B Testing for Bandit Strategies

Compares Thompson Sampling (bandit.py) vs IDS (NewBandit.py) over multiple
simulated games to determine which strategy performs better.

Usage:
    python monte_carlo.py                     # 100 trials, summary only
    python monte_carlo.py --trials 500 -v     # 500 trials, verbose
    python monte_carlo.py --use-social        # Include simulated team choices
"""

import numpy as np
import argparse
import json
import math
from typing import List, Optional, Tuple, Dict, Any

# Import from existing modules
from simulator import generate_arm_distribution, sample_reward, compute_expected_value
from bandit import BanditGame
from NewBandit import OptimalBanditGame


# =============================================================================
# SIMPLE T-TEST (no scipy needed)
# =============================================================================

def paired_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Perform a paired t-test between two samples.

    Returns:
        (t_statistic, p_value)
    """
    diff = x - y
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    if std_diff == 0:
        return 0.0, 1.0

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1

    # Approximate p-value using normal distribution for large n
    # For small n, this is less accurate but sufficient for our purposes
    if n >= 30:
        # Use normal approximation
        p_value = 2 * (1 - normal_cdf(abs(t_stat)))
    else:
        # Use t-distribution approximation
        p_value = 2 * (1 - t_cdf(abs(t_stat), df))

    return float(t_stat), float(p_value)


def normal_cdf(x: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def t_cdf(t: float, df: int) -> float:
    """
    Approximate t-distribution CDF using normal approximation.
    Good enough for df > 5.
    """
    # For large df, t approaches normal
    if df >= 30:
        return normal_cdf(t)

    # Simple approximation for smaller df
    # Uses the fact that t_df approaches N(0,1) as df increases
    adjusted_t = t * math.sqrt((df - 2) / df) if df > 2 else t
    return normal_cdf(adjusted_t)


# =============================================================================
# SIMULATOR WRAPPER
# =============================================================================

class SimulatorWrapper:
    """Wraps simulator functions for programmatic use."""

    def __init__(self, num_arms: int = 4, max_reward: int = 10, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.num_arms = num_arms
        self.max_reward = max_reward
        self.distributions = [generate_arm_distribution(max_reward) for _ in range(num_arms)]

    def pull_arm(self, arm: int) -> int:
        """Get reward from pulling an arm."""
        return sample_reward(self.distributions[arm])

    def get_expected_values(self) -> List[float]:
        """Get ground truth expected values for each arm."""
        return [compute_expected_value(d) for d in self.distributions]

    def get_best_arm(self, forbidden: int = None) -> Tuple[int, float]:
        """Get the best arm (excluding forbidden) and its expected value."""
        evs = self.get_expected_values()
        best_arm = None
        best_ev = -float('inf')
        for i, ev in enumerate(evs):
            if i != forbidden and ev > best_ev:
                best_ev = ev
                best_arm = i
        return best_arm, best_ev


# =============================================================================
# SIMULATED TEAMS (for --use-social)
# =============================================================================

class SimulatedTeams:
    """
    Simulates 4 other teams using a follow-the-leader strategy.
    70% chance to pick best-known arm, 30% random exploration.
    """

    def __init__(self, num_teams: int = 4, num_arms: int = 4, exploit_prob: float = 0.7):
        self.num_teams = num_teams
        self.num_arms = num_arms
        self.exploit_prob = exploit_prob

        # Each team tracks their own reward history
        self.team_rewards = [{arm: [] for arm in range(num_arms)} for _ in range(num_teams)]

    def get_team_choices(self, forbidden_arms: List[int], simulator: SimulatorWrapper) -> List[int]:
        """
        Get choices for all simulated teams this round.

        Args:
            forbidden_arms: List of forbidden arm for each team (can differ)
            simulator: The simulator to pull arms from

        Returns:
            List of arm choices (one per team)
        """
        choices = []

        for team_idx in range(self.num_teams):
            forbidden = forbidden_arms[team_idx] if team_idx < len(forbidden_arms) else None

            # Get allowed arms
            allowed = [a for a in range(self.num_arms) if a != forbidden]

            if np.random.random() < self.exploit_prob and any(self.team_rewards[team_idx][a] for a in allowed):
                # Exploit: pick arm with best average reward so far
                best_arm = None
                best_avg = -float('inf')
                for arm in allowed:
                    rewards = self.team_rewards[team_idx][arm]
                    if rewards:
                        avg = np.mean(rewards)
                        if avg > best_avg:
                            best_avg = avg
                            best_arm = arm

                if best_arm is not None:
                    choice = best_arm
                else:
                    choice = np.random.choice(allowed)
            else:
                # Explore: random choice
                choice = np.random.choice(allowed)

            choices.append(choice)

            # Simulate pulling the arm and record result
            reward = simulator.pull_arm(choice)
            self.team_rewards[team_idx][choice].append(reward)

        return choices


# =============================================================================
# GAME RUNNING FUNCTIONS
# =============================================================================

def run_single_game(
    strategy,
    simulator: SimulatorWrapper,
    forbidden_sequence: List[int],
    team_choices_sequence: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    """
    Run one complete game with a strategy against a simulator.

    Args:
        strategy: BanditGame or OptimalBanditGame instance
        simulator: SimulatorWrapper with pre-generated distributions
        forbidden_sequence: List of forbidden arm per round
        team_choices_sequence: Optional list of team choices per round (for social learning)

    Returns:
        Dictionary with total_reward, history, arm_stats
    """
    total_reward = 0
    num_rounds = len(forbidden_sequence)

    for round_num in range(num_rounds):
        forbidden = forbidden_sequence[round_num]

        # Record team choices from previous round (if available)
        if team_choices_sequence and round_num > 0:
            strategy.record_team_choices(team_choices_sequence[round_num - 1])

        # Get recommendation
        if hasattr(strategy, 'recommend_ids'):
            arm, _ = strategy.recommend_ids(forbidden)
        else:
            arm, _ = strategy.recommend(forbidden)

        # Get reward from simulator
        reward = simulator.pull_arm(arm)
        total_reward += reward

        # Update strategy beliefs
        strategy.record_result(arm, reward, forbidden)

    return {
        'total_reward': total_reward,
        'history': list(strategy.round_history),
        'arm_stats': strategy.get_arm_stats(),
        'arm_evs': simulator.get_expected_values()
    }


def run_trial(
    seed: int,
    num_rounds: int = 25,
    num_arms: int = 4,
    max_reward: int = 10,
    use_social: bool = False
) -> Dict[str, Any]:
    """
    Run a single trial comparing both strategies on identical conditions.

    Args:
        seed: Random seed for reproducibility
        num_rounds: Number of rounds per game
        num_arms: Number of arms
        max_reward: Maximum reward value
        use_social: Whether to include simulated team choices

    Returns:
        Dictionary with results for both strategies
    """
    # Create simulator with seed (determines arm distributions)
    np.random.seed(seed)
    simulator = SimulatorWrapper(num_arms=num_arms, max_reward=max_reward, seed=None)

    # Generate forbidden arm sequence (same for both strategies)
    forbidden_sequence = [np.random.randint(0, num_arms) for _ in range(num_rounds)]

    # Generate team choices if using social learning
    team_choices_sequence = None
    if use_social:
        # Create simulated teams
        teams = SimulatedTeams(num_teams=4, num_arms=num_arms)
        team_choices_sequence = []

        for round_num in range(num_rounds):
            # Each team gets a random forbidden arm
            team_forbidden = [np.random.randint(0, num_arms) for _ in range(4)]
            choices = teams.get_team_choices(team_forbidden, simulator)
            team_choices_sequence.append(choices)

    # Run Thompson Sampling
    np.random.seed(seed + 1000000)  # Different seed for strategy randomness
    thompson = BanditGame(num_arms=num_arms, max_reward=max_reward, num_rounds=num_rounds)
    thompson_result = run_single_game(thompson, simulator, forbidden_sequence, team_choices_sequence)

    # Run IDS (reset simulator's random state for fair comparison)
    np.random.seed(seed + 2000000)
    ids = OptimalBanditGame(num_arms=num_arms, max_reward=max_reward, num_rounds=num_rounds)
    ids_result = run_single_game(ids, simulator, forbidden_sequence, team_choices_sequence)

    return {
        'seed': seed,
        'thompson': thompson_result,
        'ids': ids_result,
        'arm_evs': simulator.get_expected_values(),
        'best_arm': int(np.argmax(simulator.get_expected_values())),
        'forbidden_sequence': forbidden_sequence
    }


def run_monte_carlo(
    num_trials: int,
    num_rounds: int = 25,
    num_arms: int = 4,
    max_reward: int = 10,
    use_social: bool = False,
    verbose: bool = False,
    base_seed: int = 42
) -> Dict[str, Any]:
    """
    Run Monte Carlo comparison of both strategies.

    Args:
        num_trials: Number of trials to run
        num_rounds: Rounds per game
        num_arms: Number of arms
        max_reward: Maximum reward
        use_social: Include simulated team choices
        verbose: Print per-trial results
        base_seed: Starting seed

    Returns:
        Dictionary with aggregated results
    """
    trials = []
    thompson_rewards = []
    ids_rewards = []

    for trial_num in range(num_trials):
        seed = base_seed + trial_num
        result = run_trial(
            seed=seed,
            num_rounds=num_rounds,
            num_arms=num_arms,
            max_reward=max_reward,
            use_social=use_social
        )
        trials.append(result)

        t_reward = result['thompson']['total_reward']
        i_reward = result['ids']['total_reward']
        thompson_rewards.append(t_reward)
        ids_rewards.append(i_reward)

        if verbose:
            winner = "IDS" if i_reward > t_reward else ("Thompson" if t_reward > i_reward else "Tie")
            diff = i_reward - t_reward
            print(f"  Trial {trial_num + 1:3d}: Thompson={t_reward:3d}  IDS={i_reward:3d}  "
                  f"Winner={winner:<8}  Diff={diff:+d}")

    # Compute statistics
    thompson_rewards = np.array(thompson_rewards)
    ids_rewards = np.array(ids_rewards)

    # Paired t-test
    t_stat, p_value = paired_ttest(ids_rewards, thompson_rewards)

    # Win counts
    ids_wins = np.sum(ids_rewards > thompson_rewards)
    thompson_wins = np.sum(thompson_rewards > ids_rewards)
    ties = np.sum(ids_rewards == thompson_rewards)

    # Margin analysis
    diffs = ids_rewards - thompson_rewards
    ids_better_mask = diffs > 0
    thompson_better_mask = diffs < 0

    return {
        'num_trials': num_trials,
        'num_rounds': num_rounds,
        'use_social': use_social,
        'thompson': {
            'mean': float(np.mean(thompson_rewards)),
            'std': float(np.std(thompson_rewards)),
            'min': int(np.min(thompson_rewards)),
            'max': int(np.max(thompson_rewards)),
            'wins': int(thompson_wins)
        },
        'ids': {
            'mean': float(np.mean(ids_rewards)),
            'std': float(np.std(ids_rewards)),
            'min': int(np.min(ids_rewards)),
            'max': int(np.max(ids_rewards)),
            'wins': int(ids_wins)
        },
        'ties': int(ties),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'ids_avg_margin_when_better': float(np.mean(diffs[ids_better_mask])) if ids_wins > 0 else 0,
        'thompson_avg_margin_when_better': float(np.mean(-diffs[thompson_better_mask])) if thompson_wins > 0 else 0,
        'trials': trials if verbose else None
    }


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_summary(results: Dict[str, Any]):
    """Print summary statistics."""
    print(f"\nMonte Carlo A/B Test Results ({results['num_trials']} trials, {results['num_rounds']} rounds each)")
    if results['use_social']:
        print("(With simulated team choices for social learning)")
    print("=" * 64)

    t = results['thompson']
    i = results['ids']

    print(f"\n{'':20} {'Thompson Sampling':>20} {'IDS (NewBandit)':>20}")
    print(f"{'':20} {'-' * 20} {'-' * 20}")
    print(f"{'Mean Total Reward:':<20} {t['mean']:>20.1f} {i['mean']:>20.1f}")
    print(f"{'Std Dev:':<20} {t['std']:>20.1f} {i['std']:>20.1f}")
    print(f"{'Min:':<20} {t['min']:>20d} {i['min']:>20d}")
    print(f"{'Max:':<20} {t['max']:>20d} {i['max']:>20d}")
    print(f"\n{'Win Rate:':<20} {t['wins']/results['num_trials']*100:>19.0f}% {i['wins']/results['num_trials']*100:>19.0f}%")

    if results['ties'] > 0:
        print(f"{'Ties:':<20} {results['ties']:>20d}")

    # Statistical significance
    p = results['p_value']
    if p < 0.001:
        sig = "p < 0.001"
    elif p < 0.01:
        sig = f"p = {p:.3f}"
    elif p < 0.05:
        sig = f"p = {p:.2f}"
    else:
        sig = f"p = {p:.2f} (not significant)"

    print(f"\nStatistical Significance: {sig} (paired t-test)")

    # Winner summary
    if p < 0.05:
        if results['ids']['mean'] > results['thompson']['mean']:
            print(f"\n>>> IDS outperforms Thompson Sampling by {results['ids']['mean'] - results['thompson']['mean']:.1f} points on average")
        else:
            print(f"\n>>> Thompson Sampling outperforms IDS by {results['thompson']['mean'] - results['ids']['mean']:.1f} points on average")
    else:
        print("\n>>> No statistically significant difference between strategies")


def print_verbose(results: Dict[str, Any]):
    """Print verbose per-trial breakdown."""
    print("\n" + "=" * 64)
    print("Distribution of Wins:")
    print("=" * 64)

    i = results['ids']
    t = results['thompson']

    if i['wins'] > 0:
        print(f"  IDS better:      {i['wins']:3d} trials (avg margin: +{results['ids_avg_margin_when_better']:.1f})")
    if t['wins'] > 0:
        print(f"  Thompson better: {t['wins']:3d} trials (avg margin: +{results['thompson_avg_margin_when_better']:.1f})")
    if results['ties'] > 0:
        print(f"  Ties:            {results['ties']:3d} trials")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo A/B Testing for Bandit Strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monte_carlo.py                     # 100 trials, summary only
  python monte_carlo.py --trials 500 -v     # 500 trials, verbose
  python monte_carlo.py --use-social        # Include simulated teams
  python monte_carlo.py -v --output results.json
        """
    )
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of trials to run (default: 100)')
    parser.add_argument('--rounds', type=int, default=25,
                        help='Rounds per game (default: 25)')
    parser.add_argument('--arms', type=int, default=4,
                        help='Number of arms (default: 4)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show per-trial breakdown')
    parser.add_argument('--use-social', action='store_true',
                        help='Enable simulated team choices for social learning')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')

    args = parser.parse_args()

    print("Running Monte Carlo A/B Test...")
    print(f"  Trials: {args.trials}")
    print(f"  Rounds per game: {args.rounds}")
    print(f"  Social learning: {'Yes' if args.use_social else 'No'}")

    if args.verbose:
        print("\nPer-Trial Results:")
        print("-" * 64)

    results = run_monte_carlo(
        num_trials=args.trials,
        num_rounds=args.rounds,
        num_arms=args.arms,
        use_social=args.use_social,
        verbose=args.verbose,
        base_seed=args.seed
    )

    print_summary(results)

    if args.verbose:
        print_verbose(results)

    if args.output:
        # Remove trials data if not verbose (too large)
        output_results = results.copy()
        if not args.verbose:
            output_results['trials'] = None

        with open(args.output, 'w') as f:
            json.dump(output_results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
