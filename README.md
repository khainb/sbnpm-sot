# sot-gmm
Official implementation for paper: Summarizing Bayesian Nonparametric Mixture Posterior - Sliced Optimal Transport Metrics for Gaussian Mixtures

# Requirements

[Python 3.11.4 (with pip)](https://www.python.org/downloads/release/python-3114/)  and [R 4.3.1](https://cran.r-project.org/bin/windows/base/old/4.3.2/) were used to produce results, however, they are not strictly required for running the code.

## Python packages
These packages were used to produce results, however, their mentioned specific version are not strictly required for running the code.
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
If the above versions of packages are not available for your system.
```
pip install torch torchvision torchaudio
pip install -r requirements_general.txt
```
## R packages
```
salso
```

# What is included?

* Implementation of Mixed Sliced Wasserstein (Mix-SW) distance, Sliced Mixture Wasserstein (SMix-W) distance, and posterior summarization framework for Dirichlet Process mixture model.
* Code for simulation
* Code for analysis on Faithful dataset
* Code for testing Monte Carlo approximation

#  Mix-SW and SMix-W
Please check out [sot_gms.py](libs%2Fsot_gms.py).

# Simulation
Code is in the folder [simulation](simulation).

* [run_truncated_dpgmm.py](simulation%2Frun_truncated_dpgmm.py): This file runs posterior inference for truncated Dirichlet process Gaussian mixture model.
* [summarizing_with_sw.py](simulation%2Fsummarizing_with_sw.py): This file runs mixing measure summarization, partition summarization, and density summarization  with vectorized SW.
* [summarizing_with_mixsw.py](simulation%2Fsummarizing_with_mixsw.py): This file runs mixing measure summarization, partition summarization, and density summarization with Mix-SW.
* [summarizing_with_smixw.py](simulation%2Fsummarizing_with_smixw.py): This file runs mixing measure summarization, partition summarization, and density summarization  with SMix-W.
* [summarizing_partition_with_salso.R](simulation%2Fsummarizing_partition_with_salso.R): This file runs partition summarization with Binder, VI, and omARI, and also evaluate them. 
* [density_from_partition.py](simulation%2Fdensity_from_partition.py): This file runs density summarization from the given partition summarization with Binder, VI, and omARI.
* [evaluate_partition_with_salso.R](simulation%2Fevaluate_partition_with_salso.R): This file evaluates partition summarization from SW, Mix-SW, and SMix-W.
* [evaluate_density.py](simulation%2Fevaluate_density.py): This file evaluates the density summarization from all methods.
* [evaluate_mixing_measures.py](simulation%2Fevaluate_mixing_measures.py): This file evaluates the mixing measure summarization from SW, Mix-SW, and SMix-W.
* [plotting_density.py](simulation%2Fplotting_density.py): This file plots the density summarization and partition summarization figures.

Please update directory path at line 3 in [summarizing_partition_with_salso.R](simulation%2Fsummarizing_partition_with_salso.R) and [evaluate_partition_with_salso.R](simulation%2Fevaluate_partition_with_salso.R).

For evaluation from provided prerun summarization:
```
cd simulation
python evaluate_density.py
python plotting_density.py
python evaluate_mixing_measures.py
Rscript summarizing_partition_with_salso.R
Rscript evaluate_partition_with_salso.R
```

For running from scratch:
```
cd simulation
python run_truncated_dpgmm.py
python summarizing_with_sw.py
python summarizing_with_mixsw.py
python summarizing_with_smixw.py
Rscript summarizing_partition_with_salso.R
python density_from_partition.py
python evaluate_density.py
python plotting_density.py
python evaluate_mixing_measures.py
Rscript evaluate_partition_with_salso.R
```

# Faithful dataset
Code is in the folder [faithful](faithful).

* [run_truncated_dpgmm.py](simulation%2Frun_truncated_dpgmm.py): This file runs posterior inference for truncated Dirichlet process Gaussian mixture model.
* [summarizing_with_sw.py](simulation%2Fsummarizing_with_sw.py): This file runs mixing measure summarization, partition summarization, and density summarization  with vectorized SW.
* [summarizing_with_mixsw.py](simulation%2Fsummarizing_with_mixsw.py): This file runs mixing measure summarization, partition summarization, and density summarization with Mix-SW.
* [summarizing_with_smixw.py](simulation%2Fsummarizing_with_smixw.py): This file runs mixing measure summarization, partition summarization, and density summarization  with SMix-W.
* [summarizing_partition_with_salso.R](simulation%2Fsummarizing_partition_with_salso.R): This file runs partition summarization with Binder, VI, and omARI, and also evaluate them. 
* [density_from_partition.py](simulation%2Fdensity_from_partition.py): This file runs density summarization from the given partition summarization with Binder, VI, and omARI.
* [evaluate_partition_with_salso.R](simulation%2Fevaluate_partition_with_salso.R): This file evaluates partition summarization from SW, Mix-SW, and SMix-W.
* [evaluate_density.py](simulation%2Fevaluate_density.py): This file evaluates the density summarization from all methods.
* [plotting_density.py](simulation%2Fplotting_density.py): This file plots the density summarization and partition summarization figures.

Please update directory path at line 3 in [summarizing_partition_with_salso.R](faithful%2Fsummarizing_partition_with_salso.R) and [evaluate_partition_with_salso.R](faithful%2Fevaluate_partition_with_salso.R).

For evaluation from provided prerun summarization:
```
cd simulation
python evaluate_density.py
python plotting_density.py
python evaluate_mixing_measures.py
Rscript summarizing_partition_with_salso.R
Rscript evaluate_partition_with_salso.R
```

For running from scratch:
```
cd faithful
python run_truncated_dpgmm.py
python summarizing_with_sw.py
python summarizing_with_mixsw.py
python summarizing_with_smixw.py
Rscript summarizing_partition_with_salso.R
python density_from_partition.py
python evaluate_density.py
python plotting_density.py
python evaluate_mixing_measures.py
Rscript evaluate_partition_with_salso.R
```


# Monte Carlo approximation
Code is in the folder [MonteCarlo](MonteCarlo)

```
cd MonteCarlo
python main.py
```