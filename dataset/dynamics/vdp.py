import jax.numpy as jnp
import jax.random as jr

# params from paper chackrabarthy
params_nominal = jnp.array([1.0]) 
x0_nominal = jnp.array([0.0, 0.0]) 

def f_xu(x, u, params):
    """ Duffing oscillator"""
    x1, x2 = x # position, velocity
    theta, = params
    u1, = u
    #F = gamma * jnp.cos(omega * t)
    dx1 = x2
    dx2 = theta * x2 * (1 - x1**2) - x1 + u1
    dx = jnp.array([dx1, dx2])
    return dx

# currenty unused
# def g_x(x):
#     return x

def params_fn(key):
    #params_nominal = jnp.array([-1.0, 0.25, 0.1, 2.5, 2]) 
    #params = params_nominal #  + jr.normal(key, params_nominal.shape) * 1.0
    params = jr.uniform(key, params_nominal.shape, minval=0.5, maxval=2.0)
    return params

def init_fn(key):
    x0 = x0_nominal + jr.uniform(key, x0_nominal.shape, minval=-1.0, maxval=1.0)
    return x0
