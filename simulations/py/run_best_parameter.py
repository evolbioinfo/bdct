import os

# To run locally:
# snakemake --snakefile Snakefile_balanced_bdsky_test --keep-going --cores 7 --config folder=results_balanced_bdsky num=100
# To run on maestro:
# snakemake --snakefile Snakefile_balanced_bdsky_test --config folder=results_balanced_bdsky num=100 --keep-going --cores 1 --cluster "sbatch ..."

localrules: all, combine_balanced_bdsky, summarize_parameter_performance

folder = config.get("folder", 'results_balanced_bdsky')
num = int(config.get("num", 100))
REPETITIONS = list(range(num))

# Input paths
BDSKY_PATH = "/home/alejandra/NEWSNAKE/simulations_bdsky/trees"
BDCT_PATH = "/home/alejandra/NEWSNAKE/simulations/BDCT0"

# Parameter sets for exploration
PARAMETER_SETS = {
    'conservative_balanced': {
        'min_branches': 30,
        'alpha': 0.005,
        'effect_size_threshold': 0.4,
        'min_time_fraction': 0.12,
        'extra_flags': ''
    },
    'moderate_balanced': {
        'min_branches': 25,
        'alpha': 0.01,
        'effect_size_threshold': 0.3,
        'min_time_fraction': 0.10,
        'extra_flags': ''
    },
    'liberal_balanced': {
        'min_branches': 20,
        'alpha': 0.02,
        'effect_size_threshold': 0.2,
        'min_time_fraction': 0.08,
        'extra_flags': ''
    },
    'external_focused': {
        'min_branches': 25,
        'alpha': 0.01,
        'effect_size_threshold': 0.25,
        'min_time_fraction': 0.10,
        'extra_flags': '--simple-decision'
    },
    'effect_emphasized': {
        'min_branches': 25,
        'alpha': 0.015,
        'effect_size_threshold': 0.5,
        'min_time_fraction': 0.10,
        'extra_flags': ''
    },
    'fixed_intervals': {
        'min_branches': 25,
        'alpha': 0.01,
        'effect_size_threshold': 0.3,
        'min_time_fraction': 0.10,
        'extra_flags': '--use-fixed-intervals --early-fraction 0.25 --late-fraction 0.25'
    }
}

os.makedirs('logs', exist_ok=True)
os.makedirs(folder, exist_ok=True)

rule all:
    input:
        os.path.join(folder, 'balanced_bdsky_tests.tab'),
        os.path.join(folder, 'parameter_performance_summary.tab')

rule balanced_bdsky_test_bdsky_trees:
    '''
    Balanced BDSKY-test on BD-Skyline trees with different parameter sets.
    '''
    input:
        nwk=os.path.join(BDSKY_PATH, 'final_tree.{i}.nwk')
    output:
        log=os.path.join(BDSKY_PATH, 'final_tree.{i}.balanced_bdsky_test.{param_set}'),
    params:
        mem=2000,
        name='balanced_bdsky_{param_set}_{i}',
        qos='fast',
        min_branches=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['min_branches'],
        alpha=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['alpha'],
        effect_size=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['effect_size_threshold'],
        min_time_frac=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['min_time_fraction'],
        extra_flags=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['extra_flags']
    threads: 1
    singularity: "docker://evolbioinfo/bdct:v0.1.25"
    shell:
        """
        python3 /home/alejandra/NEWSNAKE/bdct/balanced_sky_test.py \
            --nwk {input.nwk} \
            --log {output.log} \
            --min-branches {params.min_branches} \
            --alpha {params.alpha} \
            --effect-size-threshold {params.effect_size} \
            --min-time-fraction {params.min_time_frac} \
            {params.extra_flags} \
            --verbose
        """

rule balanced_bdsky_test_bdct_trees:
    '''
    Balanced BDSKY-test on BDCT trees (control) with different parameter sets.
    '''
    input:
        nwk=os.path.join(BDCT_PATH, 'tree.{i}.nwk')
    output:
        log=os.path.join(BDCT_PATH, 'tree.{i}.balanced_bdsky_test.{param_set}'),
    params:
        mem=2000,
        name='balanced_bdct_{param_set}_{i}',
        qos='fast',
        min_branches=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['min_branches'],
        alpha=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['alpha'],
        effect_size=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['effect_size_threshold'],
        min_time_frac=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['min_time_fraction'],
        extra_flags=lambda wildcards: PARAMETER_SETS[wildcards.param_set]['extra_flags']
    threads: 1
    singularity: "docker://evolbioinfo/bdct:v0.1.25"
    shell:
        """
        python3 /home/alejandra/NEWSNAKE/bdct/balanced_sky_test.py \
            --nwk {input.nwk} \
            --log {output.log} \
            --min-branches {params.min_branches} \
            --alpha {params.alpha} \
            --effect-size-threshold {params.effect_size} \
            --min-time-fraction {params.min_time_frac} \
            {params.extra_flags} \
            --verbose
        """

rule combine_balanced_bdsky:
    '''
    Combine balanced BDSKY test results for all parameter sets.
    '''
    input:
        bdsky_logs = expand(os.path.join(BDSKY_PATH, 'final_tree.{i}.balanced_bdsky_test.{param_set}'),
                           i=REPETITIONS, param_set=PARAMETER_SETS.keys()),
        bdct_logs = expand(os.path.join(BDCT_PATH, 'tree.{i}.balanced_bdsky_test.{param_set}'),
                          i=REPETITIONS, param_set=PARAMETER_SETS.keys())
    output:
        tab = os.path.join(folder, 'balanced_bdsky_tests.tab'),
    params:
        mem = 4000,
        name = 'combine_balanced_bdsky',
        qos = 'fast',
        param_sets = ' '.join(PARAMETER_SETS.keys())
    threads: 1
    singularity: "docker://evolbioinfo/pastml:v1.9.43"
    shell:
        """
        python3 /home/alejandra/NEWSNAKE/simulations/py/summary_table_balanced_bdsky.py \
            --logs {input.bdsky_logs} {input.bdct_logs} \
            --tab {output.tab} \
            --parameter-sets {params.param_sets}
        """

rule summarize_parameter_performance:
    '''
    Analyze parameter set performance and create summary.
    '''
    input:
        tab = os.path.join(folder, 'balanced_bdsky_tests.tab')
    output:
        summary = os.path.join(folder, 'parameter_performance_summary.tab'),
        plot = os.path.join(folder, 'parameter_performance_plot.png')
    params:
        mem = 2000,
        name = 'summarize_performance',
        qos = 'fast',
    threads: 1
    singularity: "docker://evolbioinfo/pastml:v1.9.43"
    shell:
        """
        python3 /home/alejandra/NEWSNAKE/simulations/py/analyze_parameter_performance.py \
            --input {input.tab} \
            --summary {output.summary} \
            --plot {output.plot}
        """

# Optional: Run only the best parameter set on all trees
rule best_parameter_full_run:
    '''
    Run the best performing parameter set on all trees.
    '''
    input:
        summary = os.path.join(folder, 'parameter_performance_summary.tab')
    output:
        best_results = os.path.join(folder, 'best_parameter_results.tab')
    params:
        mem = 4000,
        name = 'best_param_full',
        qos = 'fast',
        bdsky_path = BDSKY_PATH,
        bdct_path = BDCT_PATH,
        repetitions = num
    threads: 1
    singularity: "docker://evolbioinfo/pastml:v1.9.43"
    shell:
        """
        python3 /home/alejandra/NEWSNAKE/simulations/py/run_best_parameter.py \
            --summary {input.summary} \
            --bdsky-path {params.bdsky_path} \
            --bdct-path {params.bdct_path} \
            --output {output.best_results} \
            --repetitions {params.repetitions}
        """