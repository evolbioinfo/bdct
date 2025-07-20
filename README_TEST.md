# Instructions of the tests

## BDCT Test

- [BDCT TEST](bdct/model_distinguisher.py): code for the BDCT test
  - To run the test on a single tree, give the input: `--nwk path/to/tree_file.nwk --log path/to/output.log` , where the .nwk file is the tree in which the user wants to run the test and the log file will be the output with the test results.
- [BDCT RESULTS TABLE](simulations/py/summary_table_cherries.py): code for the creation of a table with the results of running the test on all the trees of our pipeline, `cherry_tests.tab`.
- [BDCT TEST PIPELINE](simulations/Snakefile_cherry_test): code to run the BDCT test on 100 trees with simple BD model and with BDCT.
  - To run the pipeline on the terminal, write: `snakemake --snakefile Snakefile_cherry_test --keep-going --cores number_of_cores`. 
    - Before running the pipeline, if the files `cherry_tests.tab` , `simulations/folder_of_trees_to_test/tree.*.cherry_test.log` already exist, they need to be deleted so the pipeline can run the jobs.
## BDEI Test
- [BDEI TEST](bdct/bdei_distinguisher.py): code for the BDEI test
  - To run the test on a single tree, give the input: `--nwk path/to/tree_file.nwk --log path/to/output.log` , where the .nwk file is the tree in which the user wants to run the test and the log file will be the output with the test results.
- [BDEI RESULTS TABLE](simulations/py/summary_table_bdei.py): code for the creation of a table with the results of running the test on all the trees of our pipeline, `bdei_tests.tab`.
- [BDEI TEST PIPELINE](simulations/Snakefile_bdei_test): code to run the BDEI test on 100 trees with simple BD model and with BDEI.
  - To run the pipeline on the terminal, write: `snakemake --snakefile Snakefile_bdei_test --keep-going --cores number_of_cores`. 
    - Before running the pipeline, if the files `bdei_tests.tab`, `simulations/BDCT0/tree.*.bdei_test.log`, `simulations/BDEICT0/tree.*.bdei_test.log` already exist, they need to be deleted so the pipeline can run the jobs.
## BDSKY Test
- [BDSKY TEST](bdct/bdei_distinguisher.py): code for the BDSKY test
  - To run the test on a single tree, give the input: `--nwk path/to/tree_file.nwk --log path/to/output.log` , where the .nwk file is the tree in which the user wants to run the test and the log file will be the output with the test results.
- [BDSKY RESULTS TABLE](simulations/py/summary_table_bdsky.py): code for the creation of a table with the results of running the test on all the trees of our pipeline, `bdsky_tests.tab`.
- [BDSKY TEST PIPELINE](simulations/Snakefile_bdsky_test): code to run the BDSKY test on 100 trees with simple BD model and with BDSKY.
  - To run the pipeline on the terminal, write: `snakemake --snakefile Snakefile_bdsky_test --keep-going --cores number_of_cores`. 
    - Before running the pipeline, if the files `bdsky_tests.tab`, `simulations/BDCT0/tree.*.bdsky_test.log`, `simulations_bdsky/trees/final_tree.*.bdei_test` already exist, they need to be deleted so the pipeline can run the jobs.
## Tree Generators
- [BDSKY TREES GENERATOR](simulations_bdsky/Snakefile_simulate): code to generate the 100 BDSKY trees with 2 intervals. 
  - To run the trees generator in the terminal, type `snakemake --snakefile Snakefile_simulate --keep-going --cores number_of_cores`
    - To change the trees' size, change the values of `min_tips` and `max_tips` in the script.
- [GENERAL TREES GENERATOR](simulations/Snakefile_simulate): code to generate any other model 100 trees.
  - To run the trees generator in the terminal, type `snakemake --snakefile Snakefile_simulate --keep-going --cores number_of_cores`
      - To change the trees' size, change the values of `m` and `M` in the script.