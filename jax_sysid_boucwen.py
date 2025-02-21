"""
Bouc-Wen benchmark.

(C) 2025 A. Bemporad, February 21, 2025
"""

import numpy as np
from jax_sysid.utils import compute_scores
from jax_sysid.models import Model, find_best_model, LinearModel
import jax
import flax.linen as nn
from pathlib import Path
from scipy.io import loadmat
import time
from joblib import cpu_count

plotfigs = True  # set to True to plot figures

# Data generation
seed = 3  # for reproducibility of results
np.random.seed(seed)

nx = 3  # number of states
ny = 1  # number of outputs
nu = 1  # number of inputs
hidden_f = 16 # number of neurons in state-update function
hidden_g = 16 # number of neurons in output function


data_folder = "bwdataset"
data_folder = Path(data_folder)
data = loadmat(data_folder / "bw_matlab.mat")

u_train = data["u"].reshape(-1, 1)
y_train = data["y"].reshape(-1, 1)

u_train = u_train / 50.0
y_train = y_train / 7e-4

u_test = loadmat(data_folder / "uval_multisine.mat")["uval_multisine"].reshape(-1, 1)
y_test = loadmat(data_folder / "yval_multisine.mat")["yval_multisine"].reshape(-1, 1)

u_test = u_test / 50.0
y_test = y_test / 7e-4

# Perform system identification
jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations


@jax.jit
def state_fcn(x, u, params):
    A, B, C, W1, W2, W3, b1, b2, W4, W5, b3, b4 = params
    return A@x+B@u+W3@nn.gelu(W1@x+W2@u+b1)+b2


@jax.jit
def output_fcn(x, u, params):
    A, B, C, W1, W2, W3, b1, b2, W4, W5, b3, b4 = params
    return C@x+W5@nn.gelu(W4@x+b3)+b4


model = Model(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn)

nnx = hidden_f 
nny = hidden_g 

def init_fcn(seed):
    np.random.seed(seed)

    if 1:
        # Initialize linear model matrices using SYSID
        lin_model = LinearModel(nx, ny, nu, feedthrough=False)
        lin_model.loss(rho_x0=1.e-3, rho_th=1.e-2)
        lin_model.optimization(adam_epochs=0, lbfgs_epochs=1000)
        lin_model.fit(y_train, u_train)
        A,B,C = lin_model.params
    else:    
        A = (np.random.rand(1)*.4+.5)*np.eye(nx)
        B = 0.1*np.random.randn(nx, nu)
        C = 0.1*np.random.randn(ny, nx)

    W1 = 0.1*np.random.randn(nnx, nx)
    W2 = 0.1*np.random.randn(nnx, nu)
    W3 = 0.1*np.random.randn(nx, nnx)
    b1 = np.zeros(nnx)
    b2 = np.zeros(nx)
    W4 = 0.1*np.random.randn(nny, nx)
    W5 = 0.1*np.random.randn(ny, nny)
    b3 = np.zeros(nny)
    b4 = np.zeros(ny)
    return [A, B, C, W1, W2, W3, b1, b2, W4, W5, b3, b4]

# initialize model coefficients
#model.init(init_fcn(seed=0))
# L2-regularization on initial state and model coefficients
model.loss(rho_x0=1.e-8, rho_th=1.e-12)
# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=200, lbfgs_epochs=5000)

t0 = time.time()
models = model.parallel_fit(y_train, u_train, init_fcn=init_fcn, seeds=range(cpu_count()))
#model, BFR = find_best_model(models, y_test, u_test, fit='BFR') # select best model on test data
model, BFR = find_best_model(models, y_train, u_train, fit='BFR') # select best model on training data
t0 = time.time()-t0

print(f"Elapsed time: {t0} s")
yhat_train, _ = model.predict(model.x0, u_train)

# use RTS Smoother to learn x0
x0_test = model.learn_x0(u_test, y_test, RTS_epochs=10)
yhat_test, _ = model.predict(x0_test, u_test)
R2, R2_test, msg = compute_scores(
    y_train, yhat_train, y_test, yhat_test, fit='BFR')

print(msg)
# print(model.sparsity_analysis()) # only useful when tau_th>0

