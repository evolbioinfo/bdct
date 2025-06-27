import logging
import numpy as np
import scipy.stats
from ete3 import Tree
from typing import List, Tuple, Optional
import argparse

DEFAULT_MIN_BRANCHES = 50  # Much higher threshold
DEFAULT_MIN_TIME_FRACTION = 0.15  # Larger minimum intervals
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


def strict_sky_test(tree, n_branches=DEFAULT_MIN_BRANCHES, min_time_fraction=DEFAULT_MIN_TIME_FRACTION,
                    strict_alpha=0.001, require_both_significant=True):
    """
    STRICT skyline test with much more stringent criteria to minimize false positives.

    Key changes:
    - REQUIRE both internal and external tests to be significant
    - Use much stricter p-value threshold (0.001 instead of 0.05)
    - Higher branch count requirements (50+ instead of 20)
    - Larger minimum time intervals
    - Bonferroni correction applied by default

    :param tree: ete3.Tree, the tree of interest
    :param n_branches: int, minimum number of branches to define interval size (default: 50)
    :param min_time_fraction: float, minimum fraction of tree height for T (default: 0.15)
    :param strict_alpha: float, strict significance level (default: 0.001)
    :param require_both_significant: bool, require both internal and external tests significant
    :return: tuple of (evidence_found, test_results, bonferroni_evidence)
    """
    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    logging.info(f'STRICT skyline test on tree with height {tree_height:.4f}')
    logging.info(f'Requirements: {n_branches}+ branches, {min_time_fraction:.1%}+ time fraction, p < {strict_alpha}')

    if tree_height == 0:
        logging.warning("Tree height is zero, cannot perform SKY test.")
        return False, None, False

    results = {}
    min_time_abs = min_time_fraction * tree_height

    # Test internal branches with STRICT criteria
    logging.info("Testing internal branches...")
    T_internal = find_time_for_n_branches(tree, n_branches, 'internal')

    if T_internal is None:
        logging.warning(f"INSUFFICIENT internal branches (need {n_branches})")
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

        # STRICT requirement: both intervals must have sufficient branches
        if len(early_internal) >= n_branches and len(late_internal) >= n_branches:
            # Perform Mann-Whitney U test
            u_result = scipy.stats.mannwhitneyu(early_internal, late_internal, alternative='two-sided')

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
                'significant_strict': u_result.pvalue < strict_alpha,
                'significant_bonferroni': u_result.pvalue < (strict_alpha / 2)
            }

            logging.info(f"Internal branches - T={T_internal:.4f}")
            logging.info(f"  Early: {len(early_internal)}, Late: {len(late_internal)} branches")
            logging.info(f"  U={u_result.statistic:.1f}, p={u_result.pvalue:.6f}")
            logging.info(f"  Significant (strict α={strict_alpha}): {u_result.pvalue < strict_alpha}")
        else:
            logging.warning(
                f"INSUFFICIENT internal branches in intervals (early: {len(early_internal)}, late: {len(late_internal)})")
            results['internal'] = None

    # Test external branches with STRICT criteria
    logging.info("Testing external branches...")
    T_external = find_time_for_n_branches(tree, n_branches, 'external')

    if T_external is None:
        logging.warning(f"INSUFFICIENT external branches (need {n_branches})")
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

        # STRICT requirement: both intervals must have sufficient branches
        if len(early_external) >= n_branches and len(late_external) >= n_branches:
            # Perform Mann-Whitney U test
            u_result = scipy.stats.mannwhitneyu(early_external, late_external, alternative='two-sided')

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
                'significant_strict': u_result.pvalue < strict_alpha,
                'significant_bonferroni': u_result.pvalue < (strict_alpha / 2)
            }

            logging.info(f"External branches - T={T_external:.4f}")
            logging.info(f"  Early: {len(early_external)}, Late: {len(late_external)} branches")
            logging.info(f"  U={u_result.statistic:.1f}, p={u_result.pvalue:.6f}")
            logging.info(f"  Significant (strict α={strict_alpha}): {u_result.pvalue < strict_alpha}")
        else:
            logging.warning(
                f"INSUFFICIENT external branches in intervals (early: {len(early_external)}, late: {len(late_external)})")
            results['external'] = None

    # STRICT evidence determination
    available_tests = [k for k, v in results.items() if v is not None]

    if require_both_significant:
        # REQUIRE BOTH internal and external tests to be significant
        if len(available_tests) < 2:
            evidence_found = False
            bonferroni_evidence = False
            logging.info("FAILED: Both internal and external tests required, but insufficient data")
        else:
            # Check if BOTH tests are significant
            internal_sig_strict = results['internal']['significant_strict'] if results['internal'] else False
            external_sig_strict = results['external']['significant_strict'] if results['external'] else False
            internal_sig_bonf = results['internal']['significant_bonferroni'] if results['internal'] else False
            external_sig_bonf = results['external']['significant_bonferroni'] if results['external'] else False

            evidence_found = internal_sig_strict and external_sig_strict
            bonferroni_evidence = internal_sig_bonf and external_sig_bonf

            logging.info(f"STRICT evidence (both required): {evidence_found}")
            logging.info(f"  Internal significant: {internal_sig_strict}")
            logging.info(f"  External significant: {external_sig_strict}")
            logging.info(f"BONFERRONI evidence (both required): {bonferroni_evidence}")
            logging.info(f"  Internal Bonferroni: {internal_sig_bonf}")
            logging.info(f"  External Bonferroni: {external_sig_bonf}")
    else:
        # Original logic: either test can provide evidence (but still strict p-values)
        evidence_found = any(results[k]['significant_strict'] for k in available_tests if results[k])
        bonferroni_evidence = any(results[k]['significant_bonferroni'] for k in available_tests if results[k])

        significant_strict = [k for k in available_tests if results[k] and results[k]['significant_strict']]
        significant_bonf = [k for k in available_tests if results[k] and results[k]['significant_bonferroni']]

        logging.info(f"STRICT evidence (either test): {evidence_found} ({significant_strict})")
        logging.info(f"BONFERRONI evidence (either test): {bonferroni_evidence} ({significant_bonf})")

    return evidence_found, results, bonferroni_evidence


def sky_test_fixed_intervals_strict(tree, early_fraction=0.25, late_fraction=0.25,
                                    min_branches=DEFAULT_MIN_BRANCHES, strict_alpha=0.001):
    """
    STRICT fixed interval approach with stringent requirements.

    :param tree: ete3.Tree, the tree of interest
    :param early_fraction: float, fraction of tree height for early interval
    :param late_fraction: float, fraction of tree height for late interval
    :param min_branches: int, minimum branches required per interval per type
    :param strict_alpha: float, strict significance level
    :return: tuple of (evidence_found, test_results, bonferroni_evidence)
    """
    annotate_tree_with_time(tree)
    tree_height = max(getattr(node, TIME) for node in tree.traverse())

    logging.info(f'STRICT fixed intervals test - early: {early_fraction:.1%}, late: {late_fraction:.1%}')
    logging.info(f'Requirements: {min_branches}+ branches per type, p < {strict_alpha}')

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

    # STRICT test for internal branches
    if len(early_internal) >= min_branches and len(late_internal) >= min_branches:
        u_result = scipy.stats.mannwhitneyu(early_internal, late_internal, alternative='two-sided')

        results['internal'] = {
            'early_interval': (0, early_end),
            'late_interval': (late_start, tree_height),
            'early_branches': early_internal,
            'late_branches': late_internal,
            'early_count': len(early_internal),
            'late_count': len(late_internal),
            'u_statistic': u_result.statistic,
            'p_value': u_result.pvalue,
            'significant_strict': u_result.pvalue < strict_alpha,
            'significant_bonferroni': u_result.pvalue < (strict_alpha / 2)
        }

        logging.info(f"Fixed internal - Early: {len(early_internal)}, Late: {len(late_internal)}")
        logging.info(f"  U={u_result.statistic:.1f}, p={u_result.pvalue:.6f}, sig: {u_result.pvalue < strict_alpha}")
    else:
        logging.warning(f"INSUFFICIENT internal branches (early: {len(early_internal)}, late: {len(late_internal)})")
        results['internal'] = None

    # STRICT test for external branches
    if len(early_external) >= min_branches and len(late_external) >= min_branches:
        u_result = scipy.stats.mannwhitneyu(early_external, late_external, alternative='two-sided')

        results['external'] = {
            'early_interval': (0, early_end),
            'late_interval': (late_start, tree_height),
            'early_branches': early_external,
            'late_branches': late_external,
            'early_count': len(early_external),
            'late_count': len(late_external),
            'u_statistic': u_result.statistic,
            'p_value': u_result.pvalue,
            'significant_strict': u_result.pvalue < strict_alpha,
            'significant_bonferroni': u_result.pvalue < (strict_alpha / 2)
        }

        logging.info(f"Fixed external - Early: {len(early_external)}, Late: {len(late_external)}")
        logging.info(f"  U={u_result.statistic:.1f}, p={u_result.pvalue:.6f}, sig: {u_result.pvalue < strict_alpha}")
    else:
        logging.warning(f"INSUFFICIENT external branches (early: {len(early_external)}, late: {len(late_external)})")
        results['external'] = None

    # STRICT evidence determination - REQUIRE BOTH tests
    available_tests = [k for k, v in results.items() if v is not None]

    if len(available_tests) < 2:
        evidence_found = False
        bonferroni_evidence = False
        logging.info("FAILED: Both internal and external tests required for fixed intervals")
    else:
        internal_sig_strict = results['internal']['significant_strict']
        external_sig_strict = results['external']['significant_strict']
        internal_sig_bonf = results['internal']['significant_bonferroni']
        external_sig_bonf = results['external']['significant_bonferroni']

        evidence_found = internal_sig_strict and external_sig_strict
        bonferroni_evidence = internal_sig_bonf and external_sig_bonf

        logging.info(f"STRICT fixed evidence (both required): {evidence_found}")
        logging.info(f"BONFERRONI fixed evidence (both required): {bonferroni_evidence}")

    return evidence_found, results, bonferroni_evidence


def plot_strict_results(tree, results, outfile=None):
    """
    Plot branch length distributions for strict skyline test results.
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

    colors = ['lightblue', 'lightcoral']

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

        # Add test results
        sig_text = "SIGNIFICANT" if result['significant_strict'] else "NOT SIGNIFICANT"
        axes[1, i].text(0.05, 0.95,
                        f'U={result["u_statistic"]:.1f}\np={result["p_value"]:.6f}\n{sig_text}',
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
    Entry point for STRICT skyline test.
    """
    parser = argparse.ArgumentParser(description="""
STRICT BD-Skyline Test - Dramatically Reduced False Positives

This is a much more stringent version designed to minimize false positives:

STRICT CRITERIA:
- Requires 50+ branches per interval (vs 10-20 previously)
- Uses p < 0.001 significance threshold (vs 0.05 previously)  
- Requires BOTH internal and external tests to be significant
- Larger minimum time intervals (15% vs 10% of tree height)
- Bonferroni correction applied by default

APPROACHES:
1. ADAPTIVE (default): Finds time T for N branches, compares early vs late
2. FIXED (--use-fixed-intervals): Compares fixed proportional intervals

This version prioritizes SPECIFICITY over sensitivity to avoid false alarms.
""")

    parser.add_argument('--nwk', required=True, type=str,
                        help="Input tree file in Newick format")
    parser.add_argument('--log', type=str, help="Output log file")
    parser.add_argument('--plot', type=str, help="Output plot file")
    parser.add_argument('--min-branches', type=int, default=DEFAULT_MIN_BRANCHES,
                        help=f"Minimum branches per interval (default: {DEFAULT_MIN_BRANCHES})")
    parser.add_argument('--min-time-fraction', type=float, default=DEFAULT_MIN_TIME_FRACTION,
                        help=f"Minimum time fraction for intervals (default: {DEFAULT_MIN_TIME_FRACTION})")
    parser.add_argument('--strict-alpha', type=float, default=0.001,
                        help="Strict significance level (default: 0.001)")
    parser.add_argument('--allow-either-test', action='store_true',
                        help="Allow either internal OR external test (default: require BOTH)")
    parser.add_argument('--use-fixed-intervals', action='store_true',
                        help="Use fixed proportional intervals instead of adaptive")
    parser.add_argument('--early-fraction', type=float, default=0.25,
                        help="Early interval fraction for fixed approach (default: 0.25)")
    parser.add_argument('--late-fraction', type=float, default=0.25,
                        help="Late interval fraction for fixed approach (default: 0.25)")
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

        # Run STRICT test
        if args.use_fixed_intervals:
            evidence_found, results, bonferroni_evidence = sky_test_fixed_intervals_strict(
                tree, args.early_fraction, args.late_fraction, args.min_branches, args.strict_alpha)
        else:
            evidence_found, results, bonferroni_evidence = strict_sky_test(
                tree, args.min_branches, args.min_time_fraction, args.strict_alpha,
                not args.allow_either_test)

        # Print results
        if bonferroni_evidence:
            print("STRICT SKYLINE test: Evidence of BD-Skyline model detected (Bonferroni corrected)")
        elif evidence_found:
            print("STRICT SKYLINE test: Evidence of BD-Skyline model detected (strict α=0.001)")
        else:
            print("STRICT SKYLINE test: No evidence of BD-Skyline model (consistent with constant-rate BD)")

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
                print(f"  Significant (α={args.strict_alpha}): {result['significant_strict']}")
                print(f"  Significant (Bonferroni): {result['significant_bonferroni']}")

        # Generate plot if requested
        if args.plot and results:
            plot_strict_results(tree, results, args.plot)

        # Write log if requested
        if args.log:
            with open(args.log, 'w') as f:
                f.write('STRICT BD-Skyline Test Results\n')
                f.write('=================================\n')
                f.write(f'Total tips in tree: {total_tips}\n')
                f.write(
                    f'Evidence of skyline model (strict α={args.strict_alpha}): {"Yes" if evidence_found else "No"}\n')
                f.write(f'Evidence of skyline model (Bonferroni): {"Yes" if bonferroni_evidence else "No"}\n')

                f.write(f'\nStrict Parameters:\n')
                f.write(f'  Minimum branches per interval: {args.min_branches}\n')
                f.write(f'  Minimum time fraction: {args.min_time_fraction}\n')
                f.write(f'  Strict significance level: {args.strict_alpha}\n')
                f.write(f'  Require both tests: {not args.allow_either_test}\n')
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
                            f.write(f'  Significant (strict): {result["significant_strict"]}\n')
                            f.write(f'  Significant (Bonferroni): {result["significant_bonferroni"]}\n')

    except Exception as e:
        logging.error(f"Error running STRICT skyline test: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())