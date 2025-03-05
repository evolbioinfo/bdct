# bdct

Maximum likelihood estimator of BD and BD-CT(1) model parameters from phylogenetic trees and a non-parametric CT detection test. 

## Preprint

Anna Zhukova, Olivier Gascuel. Accounting for partner notification in epidemiological birth-death-models. medRxiv 2024.09.09.24313296; doi:[10.1101/2024.09.09.24313296](https://doi.org/10.1101/2024.09.09.24313296)

[![DOI:10.1101/2024.09.09.24313296](https://zenodo.org/badge/DOI/10.1101/2024.09.09.24313296.svg)](https://doi.org/10.1101/2024.09.09.24313296)
[![GitHub release](https://img.shields.io/github/v/release/evolbioinfo/bdct.svg)](https://github.com/evolbioinfo/bdct/releases)
[![PyPI version](https://badge.fury.io/py/bdct.svg)](https://pypi.org/project/bdct/)
[![PyPI downloads](https://shields.io/pypi/dm/bdct)](https://pypi.org/project/bdct/)
[![Docker pulls](https://img.shields.io/docker/pulls/evolbioinfo/bdct)](https://hub.docker.com/r/evolbioinfo/bdct/tags)

## BD-CT(1) model

BD-CT(1) model extends the classical birth-death (BD) model with incomplete sampling [[Stadler 2009]](https://pubmed.ncbi.nlm.nih.gov/19631666/), by adding contact tracing (CT).
Under this model, infected individuals can transmit their pathogen with a constant rate λ, 
get removed (become non-infectious) with a constant rate ψ, 
and their pathogen can be sampled upon removal 
with a constant probability ρ. On top of that, in the BD-CT(1) model, 
at the moment of sampling the sampled individual 
might notify their most recent contact with a constant probability υ. 
Upon notification, the contact is removed almost instantaneously and their pathogen is systematically sampled upon removal 
(modeled via a constant notified sampling rate φ >> ψ).

BD-CT(1) model therefore has 5 parameters:
* λ -- transmission rate
* ψ -- removal rate
* ρ -- sampling probability upon removal
* υ -- probability to notify contacts upon sampling
* φ -- notified contact removal and sampling rate: φ >> ψ<sub>i</sub> ∀i (1 ≤ i ≤ m). The pathogen of a notified contact is sampled automatically (with a probability of 1) upon removal. 

These parameters can be expressed in terms of the following epidemiological parameters:
* R<sub>0</sub>=λ/ψ -- reproduction number
* 1/ψ -- infectious time
* 1/φ -- notified contact removal time

BD-CT(1) model makes 3 assumptions:
1. only observed individuals can notify (instead of any removed individual);
2. notified contacts are always observed upon removal;
3. only the most recent contact can get notified.

For identifiability, BD-CT(1) model requires one of the three BD model parameters (λ, ψ, ρ) to be fixed.

## BD-CT(1) parameter estimator

The bdct package provides a classical BD and a BD-CT(1) model maximum-likelihood parameter estimator 
from a user-supplied time-scaled phylogenetic tree (or a forest of trees). 
User must also provide a value for one of the three BD model parameters (λ, ψ, or ρ). 
We recommend providing the sampling probability ρ, 
which could be estimated as the number of tree tips divided by the number of declared cases for the same time period.

## CT test

The bdct package provides a non-parametric test detecting presence/absence of contact tracing in a user-supplied time-scaled phylogenetic tree/forest. 
It outputs a p-value and the number of cherries in the tree. 

## Input data
One needs to supply a time-scaled phylogenetic tree in newick format, or a collection of trees (one per line in the newick file), 
which will be treated as all belonging to the same epidemic (i.e., having the same BD-CT(1) model parameters). 
In the examples below we will use an HIV tree reconstructed from 200 sequences, 
published in [[Rasmussen _et al._ PLoS Comput. Biol. 2017]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005448), 
which you can find at [PairTree GitHub](https://github.com/davidrasm/PairTree) 
and in [hiv_zurich/Zurich.nwk](hiv_zurich/Zurich.nwk). 

## Installation

There are 4 alternative ways to run __bdct__ on your computer: 
with [docker](https://www.docker.com/community-edition), 
[apptainer](https://apptainer.org/),
in Python3, or via command line (requires installation with Python3).



### Run in python3 or command-line (for linux systems, recommended Ubuntu 21 or newer versions)

You could either install python (version 3.10 or higher) system-wide and then install bdct via pip:
```bash
sudo apt install -y python3 python3-pip python3-setuptools python3-distutils
pip3 install bdct
```

or alternatively, you could install python (version 3.10 or higher) and bdct via [conda](https://conda.io/docs/) (make sure that conda is installed first). 
Here we will create a conda environment called _bdctenv_:
```bash
conda create --name bdctenv python=3.10
conda activate bdctenv
pip install bdct
```


#### Basic usage in a command line
If you installed __bdct__ in a conda environment (here named _bdctenv_), do not forget to first activate it, e.g.

```bash
conda activate bdctenv
```

Run the following commands to check for the presence of contact tracing and estimate BD-CT(1) model parameters.
The first command applies the CT test to a tree Zurich.nwk and saves the CT-test value to the file cherry_test.txt
The second command estimated the BD-CT(1) parameters and their 95% CIs for this tree, assuming the sampling probability of 0.25, 
and saves the estimated parameters to a comma-separated file estimates_bdct.csv.
The third command estimated the classical BD parameters and their 95% CIs for this tree, assuming the sampling probability of 0.25, 
and saves the estimated parameters to a comma-separated file estimates_bd.csv.
```bash
ct_test --nwk Zurich.nwk --log cherry_test.txt
bdct_infer --nwk Zurich.nwk --ci --p 0.25 --log estimates_bdct.csv
bd_infer --nwk Zurich.nwk --ci --p 0.25 --log estimates_bd.csv
```

#### Help

To see detailed options, run:
```bash
ct_test --help
bdct_infer --help
bd_infer --help
```


### Run with docker

#### Basic usage
Once [docker](https://www.docker.com/community-edition) is installed, 
run the following command to estimate BD-CT(1) model parameters:
```bash
docker run -v <path_to_the_folder_containing_the_tree>:/data:rw -t evolbioinfo/bdct --nwk /data/Zurich.nwk --ci --p 0.25 --log /data/estimates.csv
```

This will produce a comma-separated file estimates.csv in the <path_to_the_folder_containing_the_tree> folder,
 containing the estimated parameter values and their 95% CIs (can be viewed with a text editor, Excel or Libre Office Calc).

#### Help

To see advanced options, run
```bash
docker run -t evolbioinfo/bdct -h
```



### Run with apptainer

#### Basic usage
Once [apptainer](https://apptainer.org/docs/user/latest/quick_start.html#installation) is installed, 
run the following command to estimate BD-CT(1) model parameters (from the folder where the Zurich.nwk tree is contained):

```bash
apptainer run docker://evolbioinfo/bdct --nwk Zurich.nwk --ci --p 0.25 --log estimates.csv
```

This will produce a comma-separated file estimates.csv,
 containing the estimated parameter values and their 95% CIs (can be viewed with a text editor, Excel or Libre Office Calc).


#### Help

To see advanced options, run
```bash
apptainer run docker://evolbioinfo/bdct -h
```


