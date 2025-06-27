import logging
import numpy as np
import scipy.stats
from ete3 import Tree
from typing import List, Tuple, Optional
import argparse

# BALANCED parameters - middle ground between liberal and strict
DEFAULT_MIN_BRANCHES = 25  # Between 10 and 50
DEFAULT_MIN_TIME_FRACTION = 0.10  # Slightly less restrictive than 0.15
DEFAULT_ALPHA = 0.01  # Between 0.05 and 0.001
DEFAULT_EFFECT_SIZE_THRESHOLD = 0.3  # Cohen's d threshold for meaningful difference
TIME = 'time'


def annotate_tree_with_time(tree):
    """
    Annotates tree nodes with their time from the root.

    :param tree: ete3.Tree, the tree to annotate
    """
    # Set root time to 0
    tree.add_features(time=0.0)

    # Calculate times for all nodes
    for node in tree.traverse("preorder"):
        if not node.is_root():
            node.add_features(time=node.up.time + node.dist)


def extract_branches_in_interval(tree, t_start, t_end):
    """
    Extract branch lengths that fall completely within a time interval.

    :param tree: ete3.Tree, the tree of interest
    :param t_start: float, start time of interval
    :param t_end: float, end time of interval
    :return: tuple of (internal_branches, external_branches)
    """
    internal_branches = []
    external_branches = []

    for node in tree.traverse():
        if node.is_root() or node.dist is None:
            continue

        # Calculate branch start and end times
        branch_end_time = getattr(node, TIME)
        branch_start_time = branch_end_time - node.dist

        # Check if branch falls completely within interval
        if branch_start_time >= t_start and branch_end_time <= t_end:
            if node.is_leaf():
                external_branches.append(node.dist)
            else:
                internal_branches.append(node.dist)

    return internal_branches, external_branches


def find_time_for_n_branches(tree, n_branches, branch_type='internal'):
    """
    Find the time T from root needed to accumulate n complete branches of specified type.

    :param tree: ete3.Tree, the tree of interest
    :param n_branches: int, number of branches to accumulate
    :param branch_type: str, either 'internal' or 'external'
    :return: float, time T from root, or None if not enough branches found
    """
    # Collect all branches with their end times
    branches_with_times = []

    for node in tree.traverse():
        if node.is_root() or node.dist is None:
            continue

        branch_end_time = getattr(node, TIME)
        branch_start_time = branch_end_time - node.dist

        # Check branch type
        if branch_type == 'internal' and not node.is_leaf():
            branches_with_times.append((branch_end_time, branch_start_time, node.dist))
        elif branch_type == 'external' and node.is_leaf():
            branches_with_times.append((branch_end_time, branch_start_time, node.dist))

    if len(branches_with_times) < n_branches:
        return None

    # Sort by end time to process branches chronologically
    branches_with_times.sort(key=lambda x: x[0])

    # Find the time when we have n complete branches
    complete_branches = 0
    current_time = 0.0

    for branch_end_time, branch_start_time, branch_length in branches_with_times:
        # Check if this branch would be complete by its end time
        if branch_start_time >= 0:  # Branch starts at or after root
            complete_branches += 1
            if complete_branches >= n_branches:
                return branch_end_time

    return None


def calculate_effect_size(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.

    :param group1: list, first group of values
    :param group2: list, second group of values
    :return: float, Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return abs(mean1 - mean2) / pooled_std


def balanced_sky_test(tree, min_branches=DEFAULT_MIN_BRANCHES, min_time_fraction=DEFAULT_MIN_TIME_FRACTION,
                      alpha=DEFAULT_ALPHA, effect_size_threshold=DEFAULT_EFFECT_SIZE_THRESHOLD,
                      smart_decision=True):
    """
    BALANCED skyline test that finds middle ground between liberal and strict approaches.

    Key features:
    - Moderate branch count requirements (25 vs 10/50)
    - Moderate significance threshold (0.01 vs 0.05/0.001)
    - Smart decision logic: requires strong evidence from at least one test,
      OR moderate evidence from both tests
    - Effect size consideration in addition to p-values
    - Bonferroni correction available but not overly restrictive

    :param tree: ete3.Tree, the tree of interest
    :param min_branches: int, minimum number of branches to define interval size
    :param min_time_fraction: float, minimum fraction of tree height for T
    :param alpha: float, significance level
    :param effect_size_threshold: float, minimum Cohen's d for meaningful difference
    :param smart_decision: bool, use smart decision logic vs simple OR logic
    :return: tuple of (evidence_found, test_results, bonferroni_evidence)
    """
    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    logging.info(f'BALANCED skyline test on tree with height {tree_height:.4f}')
    logging.info(
        f'Parameters: {min_branches}+ branches, {min_time_fraction:.1%}+ time, α={alpha}, effect size ≥{effect_size_threshold}')

    if tree_height == 0:
        logging.warning("Tree height is zero, cannot perform SKY test.")
        return False, None, False

    results = {}
    min_time_abs = min_time_fraction * tree_height

    # Test internal branches
    logging.info("Testing internal branches...")
    T_internal = find_time_for_n_branches(tree, min_branches, 'internal')

    if T_internal is None:
        logging.warning(f"Insufficient internal branches (need {min_branches})")
        results['internal'] = None
    elif T_internal >= tree_height:
        logging.warning(f"Internal branch time {T_internal:.4f} exceeds tree height {tree_height:.4f}")
        results['internal'] = None
    elif T_internal < min_time_abs:
        logging.warning(f"Internal branch time {T_internal:.4f} below minimum {min_time_abs:.4f}")
        results['internal'] = None
    else:
        # Extract branches from early interval [0, T_internal]
        early_internal, _ = extract_branches_in_interval(tree, 0, T_internal)

        # Extract branches from late interval [tree_height - T_internal, tree_height]
        late_start = tree_height - T_internal
        if late_start < 0:
            late_start = 0
        late_internal, _ = extract_branches_in_interval(tree, late_start, tree_height)

        if len(early_internal) >= min_branches and len(late_internal) >= min_branches:
            # Perform Mann-Whitney U test
            u_result = scipy.stats.mannwhitneyu(early_internal, late_internal, alternative='two-sided')

            # Calculate effect size
            effect_size = calculate_effect_size(early_internal, late_internal)

            results['internal'] = {
                'T': T_internal,
                'early_interval': (0, T_internal),
                'late_interval': (late_start, tree_height),
                'early_branches': early_internal,
                'late_branches': late_internal,
                'early_count': len(early_internal),
                'late_count': len(late_internal),
                'u_statistic': u_result.statistic,
                'p_value': u_result.pvalue,
                'effect_size': effect_size,
                'significant': u_result.pvalue < alpha,
                'large_effect': effect_size >= effect_size_threshold,
                'strong_evidence': u_result.pvalue < alpha and effect_size >= effect_size_threshold,
                'bonferroni_significant': u_result.pvalue < (alpha / 2)
            }

            logging.info(f"Internal branches - T={T_internal:.4f}")
            logging.info(f"  Early: {len(early_internal)}, Late: {len(late_internal)} branches")
            logging.info(f"  U={u_result.statistic:.1f}, p={u_result.pvalue:.6f}, d={effect_size:.3f}")
            logging.info(
                f"  Significant: {u_result.pvalue < alpha}, Large effect: {effect_size >= effect_size_threshold}")
        else:
            logging.warning(
                f"Insufficient internal branches in intervals (early: {len(early_internal)}, late: {len(late_internal)})")
            results['internal'] = None

    # Test external branches
    logging.info("Testing external branches...")
    T_external = find_time_for_n_branches(tree, min_branches, 'external')

    if T_external is None:
        logging.warning(f"Insufficient external branches (need {min_branches})")
        results['external'] = None
    elif T_external >= tree_height:
        logging.warning(f"External branch time {T_external:.4f} exceeds tree height {tree_height:.4f}")
        results['external'] = None
    elif T_external < min_time_abs:
        logging.warning(f"External branch time {T_external:.4f} below minimum {min_time_abs:.4f}")
        results['external'] = None
    else:
        # Extract branches from early interval [0, T_external]
        _, early_external = extract_branches_in_interval(tree, 0, T_external)

        # Extract branches from late interval [tree_height - T_external, tree_height]
        late_start = tree_height - T_external
        if late_start < 0:
            late_start = 0
        _, late_external = extract_branches_in_interval(tree, late_start, tree_height)

        if len(early_external) >= min_branches and len(late_external) >= min_branches:
            # Perform Mann-Whitney U test
            u_result = scipy.stats.mannwhitneyu(early_external, late_external, alternative='two-sided')

            # Calculate effect size
            effect_size = calculate_effect_size(early_external, late_external)

            results['external'] = {
                'T': T_external,
                'early_interval': (0, T_external),
                'late_interval': (late_start, tree_height),
                'early_branches': early_external,
                'late_branches': late_external,
                'early_count': len(early_external),
                'late_count': len(late_external),
                'u_statistic': u_result.statistic,
                'p_value': u_result.pvalue,
                'effect_size': effect_size,
                'significant': u_result.pvalue < alpha,
                'large_effect': effect_size >= effect_size_threshold,
                'strong_evidence': u_result.pvalue < alpha and effect_size >= effect_size_threshold,
                'bonferroni_significant': u_result.pvalue < (alpha / 2)
            }

            logging.info(f"External branches - T={T_external:.4f}")
            logging.info(f"  Early: {len(early_external)}, Late: {len(late_external)} branches")
            logging.info(f"  U={u_result.statistic:.1f}, p={u_result.pvalue:.6f}, d={effect_size:.3f}")
            logging.info(
                f"  Significant: {u_result.pvalue < alpha}, Large effect: {effect_size >= effect_size_threshold}")
        else:
            logging.warning(
                f"Insufficient external branches in intervals (early: {len(early_external)}, late: {len(late_external)})")
            results['external'] = None

    # BALANCED evidence determination
    available_tests = [k for k, v in results.items() if v is not None]

    if not available_tests:
        evidence_found = False
        bonferroni_evidence = False
        logging.info("No valid tests available")
    elif smart_decision:
        # SMART DECISION LOGIC:
        # Strong evidence: At least one test with both significance AND large effect size
        # OR: Both tests significant (even if effect sizes are moderate)

        strong_evidence_tests = [k for k in available_tests if results[k]['strong_evidence']]
        significant_tests = [k for k in available_tests if results[k]['significant']]
        bonferroni_tests = [k for k in available_tests if results[k]['bonferroni_significant']]

        # Evidence found if:
        # 1. At least one test has strong evidence (significant + large effect), OR
        # 2. Both available tests are significant (regardless of effect size)
        evidence_found = (len(strong_evidence_tests) >= 1 or
                          (len(available_tests) >= 2 and len(significant_tests) >= 2))

        # Bonferroni evidence: at least one test significant after correction
        bonferroni_evidence = len(bonferroni_tests) >= 1

        logging.info(f"SMART decision logic:")
        logging.info(f"  Available tests: {len(available_tests)} ({available_tests})")
        logging.info(f"  Strong evidence tests: {len(strong_evidence_tests)} ({strong_evidence_tests})")
        logging.info(f"  Significant tests: {len(significant_tests)} ({significant_tests})")
        logging.info(f"  Evidence found: {evidence_found}")
        logging.info(f"  Bonferroni evidence: {bonferroni_evidence}")

    else:
        # Simple OR logic: any significant test provides evidence
        significant_tests = [k for k in available_tests if results[k]['significant']]
        bonferroni_tests = [k for k in available_tests if results[k]['bonferroni_significant']]

        evidence_found = len(significant_tests) >= 1
        bonferroni_evidence = len(bonferroni_tests) >= 1

        logging.info(f"Simple OR logic: evidence found: {evidence_found} ({significant_tests})")
        logging.info(f"Bonferroni evidence: {bonferroni_evidence} ({bonferroni_tests})")

    return evidence_found, results, bonferroni_evidence


def sky_test_fixed_intervals_balanced(tree, early_fraction=0.3, late_fraction=0.3,
                                      min_branches=DEFAULT_MIN_BRANCHES, alpha=DEFAULT_ALPHA,
                                      effect_size_threshold=DEFAULT_EFFECT_SIZE_THRESHOLD):
    """
    BALANCED fixed interval approach.

    :param tree: ete3.Tree, the tree of interest
    :param early_fraction: float, fraction of tree height for early interval
    :param late_fraction: float, fraction of tree height for late interval
    :param min_branches: int, minimum branches required per interval per type
    :param alpha: float, significance level
    :param effect_size_threshold: float, minimum effect size threshold
    :return: tuple of (evidence_found, test_results, bonferroni_evidence)
    """
    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    logging.info(f'BALANCED fixed intervals test - early: {early_fraction:.1%}, late: {late_fraction:.1%}')

    if tree_height == 0:
        logging.warning("Tree height is zero, cannot perform SKY test.")
        return False, None, False

    # Define intervals
    early_end = early_fraction * tree_height
    late_start = tree_height - (late_fraction * tree_height)

    logging.info(f'Early: [0, {early_end:.4f}], Late: [{late_start:.4f}, {tree_height:.4f}]')

    # Extract branches for both intervals
    early_internal, early_external = extract_branches_in_interval(tree, 0, early_end)
    late_internal, late_external = extract_branches_in_interval(tree, late_start, tree_height)

    results = {}

    # Test internal branches
    if len(early_internal) >= min_branches and len(late_internal) >= min_branches:
        u_result = scipy.stats.mannwhitneyu(early_internal, late_internal, alternative='two-sided')
        effect_size = calculate_effect_size(early_internal, late_internal)

        results['internal'] = {
            'early_interval': (0, early_end),
            'late_interval': (late_start, tree_height),
            'early_branches': early_internal,
            'late_branches': late_internal,
            'early_count': len(early_internal),
            'late_count': len(late_internal),
            'u_statistic': u_result.statistic,
            'p_value': u_result.pvalue,
            'effect_size': effect_size,
            'significant': u_result.pvalue < alpha,
            'large_effect': effect_size >= effect_size_threshold,
            'strong_evidence': u_result.pvalue < alpha and effect_size >= effect_size_threshold,
            'bonferroni_significant': u_result.pvalue < (alpha / 2)
        }

        logging.info(f"Fixed internal - Early: {len(early_internal)}, Late: {len(late_internal)}")
        logging.info(f"  U={u_result.statistic:.1f}, p={u_result.pvalue:.6f}, d={effect_size:.3f}")
    else:
        logging.warning(f"Insufficient internal branches (early: {len(early_internal)}, late: {len(late_internal)})")
        results['internal'] = None

    # Test external branches
    if len(early_external) >= min_branches and len(late_external) >= min_branches:
        u_result = scipy.stats.mannwhitneyu(early_external, late_external, alternative='two-sided')
        effect_size = calculate_effect_size(early_external, late_external)

        results['external'] = {
            'early_interval': (0, early_end),
            'late_interval': (late_start, tree_height),
            'early_branches': early_external,
            'late_branches': late_external,
            'early_count': len(early_external),
            'late_count': len(late_external),
            'u_statistic': u_result.statistic,
            'p_value': u_result.pvalue,
            'effect_size': effect_size,
            'significant': u_result.pvalue < alpha,
            'large_effect': effect_size >= effect_size_threshold,
            'strong_evidence': u_result.pvalue < alpha and effect_size >= effect_size_threshold,
            'bonferroni_significant': u_result.pvalue < (alpha / 2)
        }

        logging.info(f"Fixed external - Early: {len(early_external)}, Late: {len(late_external)}")
        logging.info(f"  U={u_result.statistic:.1f}, p={u_result.pvalue:.6f}, d={effect_size:.3f}")
    else:
        logging.warning(f"Insufficient external branches (early: {len(early_external)}, late: {len(late_external)})")
        results['external'] = None

    # Balanced decision logic for fixed intervals
    available_tests = [k for k, v in results.items() if v is not None]

    if not available_tests:
        evidence_found = False
        bonferroni_evidence = False
    else:
        strong_evidence_tests = [k for k in available_tests if results[k]['strong_evidence']]
        significant_tests = [k for k in available_tests if results[k]['significant']]
        bonferroni_tests = [k for k in available_tests if results[k]['bonferroni_significant']]

        # Same smart logic as adaptive version
        evidence_found = (len(strong_evidence_tests) >= 1 or
                          (len(available_tests) >= 2 and len(significant_tests) >= 2))
        bonferroni_evidence = len(bonferroni_tests) >= 1

    return evidence_found, results, bonferroni_evidence


def plot_balanced_results(tree, results, outfile=None):
    """
    Plot branch length distributions for balanced skyline test results.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logging.error("matplotlib and seaborn required for plotting.")
        return

    # Count valid results
    valid_results = [k for k, v in results.items() if v is not None]
    if not valid_results:
        logging.warning("No valid results to plot.")
        return

    n_plots = len(valid_results)
    fig, axes = plt.subplots(2, n_plots, figsize=(6 * n_plots, 8))

    if n_plots == 1:
        axes = axes.reshape(2, 1)

    colors = ['lightsteelblue', 'lightcoral']

    for i, branch_type in enumerate(valid_results):
        result = results[branch_type]

        # Plot early interval
        axes[0, i].hist(result['early_branches'], bins=15, alpha=0.7,
                        color=colors[0], edgecolor='black', label='Early')
        axes[0, i].set_title(f'{branch_type.capitalize()} Branches - Early\n'
                             f'[{result["early_interval"][0]:.2f}, {result["early_interval"][1]:.2f}]\n'
                             f'({result["early_count"]} branches)')
        axes[0, i].set_xlabel('Branch Length')
        axes[0, i].set_ylabel('Frequency')

        # Plot late interval
        axes[1, i].hist(result['late_branches'], bins=15, alpha=0.7,
                        color=colors[1], edgecolor='black', label='Late')
        axes[1, i].set_title(f'{branch_type.capitalize()} Branches - Late\n'
                             f'[{result["late_interval"][0]:.2f}, {result["late_interval"][1]:.2f}]\n'
                             f'({result["late_count"]} branches)')
        axes[1, i].set_xlabel('Branch Length')
        axes[1, i].set_ylabel('Frequency')

        # Add test results with effect size
        sig_text = "SIGNIFICANT" if result['significant'] else "NOT SIGNIFICANT"
        effect_text = f"LARGE EFFECT" if result['large_effect'] else "SMALL/MEDIUM EFFECT"
        axes[1, i].text(0.05, 0.95,
                        f'U={result["u_statistic"]:.1f}\np={result["p_value"]:.6f}\nd={result["effect_size"]:.3f}\n{sig_text}\n{effect_text}',
                        transform=axes[1, i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if outfile:
        try:
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            logging.info(f"Plot saved to {outfile}")
        except Exception as e:
            logging.error(f"Error saving plot to {outfile}: {e}")
    else:
        plt.show()


def main():
    """
    Entry point for BALANCED skyline test.
    """
    parser = argparse.ArgumentParser(description="""
BALANCED BD-Skyline Test - Optimized Sensitivity vs Specificity

This version finds the middle ground between the liberal and strict approaches:

BALANCED CRITERIA:
- Moderate branch requirements: 25+ per interval (vs 10/50)
- Moderate significance: p < 0.01 (vs 0.05/0.001)  
- Effect size consideration: Cohen's d ≥ 0.3 for meaningful differences
- Smart decision logic: Strong evidence from one test OR significance from both
- Minimum time intervals: 10% of tree height (vs none/15%)

DECISION LOGIC:
Evidence found if:
1. At least one test shows STRONG evidence (significant + large effect size), OR
2. Both available tests are significant (even with moderate effect sizes)

This balances sensitivity (detecting true skyline models) with specificity 
(avoiding false positives from constant-rate BD models).
""")

    parser.add_argument('--nwk', required=True, type=str,
                        help="Input tree file in Newick format")
    parser.add_argument('--log', type=str, help="Output log file")
    parser.add_argument('--plot', type=str, help="Output plot file")
    parser.add_argument('--min-branches', type=int, default=DEFAULT_MIN_BRANCHES,
                        help=f"Minimum branches per interval (default: {DEFAULT_MIN_BRANCHES})")
    parser.add_argument('--min-time-fraction', type=float, default=DEFAULT_MIN_TIME_FRACTION,
                        help=f"Minimum time fraction for intervals (default: {DEFAULT_MIN_TIME_FRACTION})")
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help=f"Significance level (default: {DEFAULT_ALPHA})")
    parser.add_argument('--effect-size-threshold', type=float, default=DEFAULT_EFFECT_SIZE_THRESHOLD,
                        help=f"Minimum effect size (Cohen's d) threshold (default: {DEFAULT_EFFECT_SIZE_THRESHOLD})")
    parser.add_argument('--simple-decision', action='store_true',
                        help="Use simple OR logic instead of smart decision logic")
    parser.add_argument('--use-fixed-intervals', action='store_true',
                        help="Use fixed proportional intervals instead of adaptive")
    parser.add_argument('--early-fraction', type=float, default=0.3,
                        help="Early interval fraction for fixed approach (default: 0.3)")
    parser.add_argument('--late-fraction', type=float, default=0.3,
                        help="Late interval fraction for fixed approach (default: 0.3)")
    parser.add_argument('--verbose', action='store_true', help="Verbose logging")

    args = parser.parse_args()

    # Set up logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    try:
        # Read tree
        tree = Tree(args.nwk, format=1)
        total_tips = len(tree.get_leaves())

        print(f"Total tips in tree: {total_tips}")

        # Run BALANCED test
        if args.use_fixed_intervals:
            evidence_found, results, bonferroni_evidence = sky_test_fixed_intervals_balanced(
                tree, args.early_fraction, args.late_fraction, args.min_branches,
                args.alpha, args.effect_size_threshold)
        else:
            evidence_found, results, bonferroni_evidence = balanced_sky_test(
                tree, args.min_branches, args.min_time_fraction, args.alpha,
                args.effect_size_threshold, not args.simple_decision)

        # Print results
        if bonferroni_evidence:
            print("BALANCED SKYLINE test: Evidence of BD-Skyline model detected (Bonferroni corrected)")
        elif evidence_found:
            print(f"BALANCED SKYLINE test: Evidence of BD-Skyline model detected (α={args.alpha})")
        else:
            print("BALANCED SKYLINE test: No evidence of BD-Skyline model (consistent with constant-rate BD)")

        # Print detailed results
        for branch_type in ['internal', 'external']:
            if results and results.get(branch_type) is not None:
                result = results[branch_type]
                if 'T' in result:
                    print(f"\n{branch_type.capitalize()} branches (adaptive):")
                    print(f"  T = {result['T']:.4f}")
                else:
                    print(f"\n{branch_type.capitalize()} branches (fixed):")
                print(
                    f"  Early interval [{result['early_interval'][0]:.4f}, {result['early_interval'][1]:.4f}]: {result['early_count']} branches")
                print(
                    f"  Late interval [{result['late_interval'][0]:.4f}, {result['late_interval'][1]:.4f}]: {result['late_count']} branches")
                print(f"  Mann-Whitney U statistic: {result['u_statistic']:.4f}")
                print(f"  p-value: {result['p_value']:.6f}")
                print(f"  Effect size (Cohen's d): {result['effect_size']:.3f}")
                print(f"  Significant (α={args.alpha}): {result['significant']}")
                print(f"  Large effect (d≥{args.effect_size_threshold}): {result['large_effect']}")
                print(f"  Strong evidence (sig + large effect): {result['strong_evidence']}")
                print(f"  Bonferroni significant: {result['bonferroni_significant']}")

        # Generate plot if requested
        if args.plot and results:
            plot_balanced_results(tree, results, args.plot)

        # Write log if requested
        if args.log:
            with open(args.log, 'w') as f:
                f.write('BALANCED BD-Skyline Test Results\n')
                f.write('===================================\n')
                f.write(f'Total tips in tree: {total_tips}\n')
                f.write(f'Evidence of skyline model (α={args.alpha}): {"Yes" if evidence_found else "No"}\n')
                f.write(f'Evidence of skyline model (Bonferroni): {"Yes" if bonferroni_evidence else "No"}\n')

                f.write(f'\nBalanced Parameters:\n')
                f.write(f'  Minimum branches per interval: {args.min_branches}\n')
                f.write(f'  Minimum time fraction: {args.min_time_fraction}\n')
                f.write(f'  Significance level: {args.alpha}\n')
                f.write(f'  Effect size threshold: {args.effect_size_threshold}\n')
                f.write(f'  Smart decision logic: {not args.simple_decision}\n')
                f.write(f'  Method: {"Fixed intervals" if args.use_fixed_intervals else "Adaptive"}\n')

                if results:
                    for branch_type in ['internal', 'external']:
                        if results.get(branch_type) is not None:
                            result = results[branch_type]
                            f.write(f'\n{branch_type.capitalize()} branches:\n')
                            if 'T' in result:
                                f.write(f'  T = {result["T"]:.6f}\n')
                            f.write(
                                f'  Early interval: [{result["early_interval"][0]:.6f}, {result["early_interval"][1]:.6f}] ({result["early_count"]} branches)\n')
                            f.write(
                                f'  Late interval: [{result["late_interval"][0]:.6f}, {result["late_interval"][1]:.6f}] ({result["late_count"]} branches)\n')
                            f.write(f'  Mann-Whitney U statistic: {result["u_statistic"]:.6f}\n')
                            f.write(f'  p-value: {result["p_value"]:.6f}\n')
                            f.write(f'  Effect size (Cohen\'s d): {result["effect_size"]:.6f}\n')
                            f.write(f'  Significant: {result["significant"]}\n')
                            f.write(f'  Large effect: {result["large_effect"]}\n')
                            f.write(f'  Strong evidence: {result["strong_evidence"]}\n')
                            f.write(f'  Bonferroni significant: {result["bonferroni_significant"]}\n')

    except Exception as e:
        logging.error(f"Error running BALANCED skyline test: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())