# Learning reduced-order neural model structures

This repository accompanies the paper *Manifold meta-learning for reduced-complexity neural system identification* by Marco Forgione, Ankush Chakrabarty, Dario Piga, Mattei Rufolo and Alberto Bemporad (Automatica, 2026)


## Main files

The main files are:

* [train_full_alldata.ipynb](train_full_alldata.ipynb): Train a single full-order model on the full training dataset
* [test_full_alldata.ipynb](test_full_alldata.ipynb): Test the full-order model
* [linear_identification.m](linear_identification.m): Linear identification baseline
* [train_mc_full.ipynb](train_mc_full.ipynb): Train full-order models on training datasets of different lengths
* [meta_train.ipynb](meta_train.ipynb): Learn the reduced-order architecture on the meta dataset
* [train_reduced_mc.ipynb](train_reduced_mc.ipynb): Train reduced-order models on sequences of different lengths
* [analyze_mc.ipynb](analyze_mc.ipynb): Analyze Monte Carlo experiments and obtain paper figures
* [train_full_and_reduced.ipynb](train_full_and_reduced.ipynb): Train a single full-order model and a single reduced-order model on a single dataset, for illustration purpose
* [meta_train_maml.ipynb](meta_train_maml.ipynb): Learn a parameter initialization on the meta dataset with MAML
* [train_maml_mc.ipynb](train_maml_mc.ipynb): Train full-order models with MAML initialization on training datasets of different lengths

It is preferrable to run the training files from the command line through ipython:

``
$ ipython 
``

``
run <file_name>
``

Additional python files:

* [ae.py](ae.py) Autoencoder model blocks
* [neuralss.py](neuralss.py) Neural state-space base architecture
* [lr.py](lr.py) Learning rate scheduling
* [plot_utils.py](plot_utils.py) Matplotlib settings for good-looking plots in papers

# Software requirements:
Simulations were performed in Python 3.10 using the following packages:

 * [jax](https://github.com/jax-ml/jax) for automatic differentiation and GPU computation
 * [flax](https://github.com/google/flax) for the encoder network
 * [optax](https://github.com/google-deepmind/optax) and [jaxopt](https://github.com/google/jaxopt) for optimization
 * [diffrax](https://github.com/patrick-kidger/diffrax) and [equinox](https://github.com/patrick-kidger/equinox) for simulating the Bouc-Wen system
 * [jax-sysid](https://github.com/bemporad/jax-sysid) for the linear baseline
 * [nonlinear_benchmarks](https://github.com/GerbenBeintema/nonlinear_benchmarks) for loading the Bouc-Wen benchmark data
 * numpy, scipy, pandas, matplotlib, seaborn, and tqdm
 * [wandb](https://wandb.ai) (optional) for logging the meta-training runs

# Citing

If you find this project useful, we encourage you to

* Star this repository :star: 
* Cite the [paper](https://arxiv.org/abs/2504.11811)

To cite the paper, you may use the following BibTeX entry:
```
@article{forgione2027manifold,
  title = {Manifold meta-learning for reduced-complexity neural system identification},
  journal = {Automatica},
  volume = {196},
  pages = {113333},
  year = {2027},
  issn = {0005-1098},
  doi = {https://doi.org/10.1016/j.automatica.2026.113333},
  url = {https://www.sciencedirect.com/science/article/pii/S0005109826005145},
  author = {Marco Forgione and Ankush Chakrabarty and Dario Piga and Matteo Rufolo and Alberto Bemporad}
}
```