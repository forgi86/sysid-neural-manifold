# Learning reduced-order neural model structures

This repository contains the Python code to reproduce the results of the paper Learning reduced-order neural model structures




## Main files

The main files to run are:

* [train_full_alldata.ipynb](train_full_alldata.ipynb): Train a single full-order model on the full training dataset
* [train_mc_full.ipynb](train_mc_full.ipynb): Train full-order models on training datasets of different lengths
* [meta_train.ipynb](meta_train.ipynb): Learn the reduced-order architecture on the data distribution (meta dataset)
* [train_mc_reduced.ipynb](train_mc_reduced.ipynb): Train reduced-order models on sequences of different lengths

* [ae.py](ae.py) Autoencoder model blocks
* [neuralss.py](neuralss.py) Neural state-space base learner model