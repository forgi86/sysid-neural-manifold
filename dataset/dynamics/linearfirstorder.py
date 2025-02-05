import jax.numpy as jnp
import jax.random as jr


def f_xu(x, u, params):
    """ Duffing oscillator"""
    a = params
    return a * x + u

# currenty unused
# def g_x(x):
#     return x

def params_fn(key):
    params_nominal = jnp.array([-0.1]) 
    params = params_nominal * jr.uniform(key, params_nominal.shape, minval=0.2, maxval=1.8)
    return params

def init_fn(key):
    x0 = jr.uniform(key, (1,), minval=-1, maxval=1)
    return x0
