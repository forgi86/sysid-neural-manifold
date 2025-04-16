# Learning reduced-order neural model structures

This repository contains the Python code to reproduce the results of the paper *Manifold meta-learning for reduced-complexity neural system identification*


## Main files

The main files are:

* [train_full_alldata.ipynb](train_full_alldata.ipynb): Train a single full-order model on the full training dataset
* [test_full_alldata.ipynb](test_full_alldata.ipynb): Test the full-order model
* [linear_identification.m](linear_identification.m): Linear identification baseline
* [train_mc_full.ipynb](train_mc_full.ipynb): Train full-order models on training datasets of different lengths
* [meta_train.ipynb](meta_train.ipynb): Learn the reduced-order architecture on the data distribution (meta dataset)
* [train_reduced_mc.ipynb](train_reduced_mc.ipynb): Train reduced-order models on sequences of different lengths
* [analyze_mc.ipynb](analyze_mc.ipynb): Analyze Monte Carlo experiments and obtain paper figures
* [train_full_and_reduced.ipynb](train_full_and_reduced.ipynb): Train a single full-order model and a single reduced-order model on a single dataset, for illustration purpose

It is preferrable to run the training files from the command line through ipython:

``
$ ipython 
``

``
run <file_name>
``

Additional python files:

* [ae.py](ae.py) Autoencoder model blocks
* [neuralss.py](neuralss.py) Neural state-space base learner model
* [lr.py](lr.py) Learning rate scheduling
* [plot_utils.py](plot_utils.py) Matplotlib settings for good-looking plots in papers
