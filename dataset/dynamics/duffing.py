import jax.numpy as jnp
import jax.random as jr

# params from wikipedia
params_nominal = jnp.array([-1.0, 0.25, 0.1, 2.5, 2]) 
x0_nominal = jnp.array([0.0, 0.0]) 

def f_xu(x, u, params):
    """ Duffing oscillator"""
    p, v = x # position, velocity
    alpha, beta, delta, gamma, omega = params
    F = u[0] # F, = u  
    #F = gamma * jnp.cos(omega * t)
    dp = v
    dv = -delta * v -alpha * p  -beta * p**3 + F
    dx = jnp.array([dp, dv])
    return dx

# currenty unused
# def g_x(x):
#     return x

def params_fn(key):
    #params_nominal = jnp.array([-1.0, 0.25, 0.1, 2.5, 2]) 
    params = params_nominal * jr.uniform(key, params_nominal.shape, minval=0.9, maxval=1.1)
    return params

def init_fn(key):
    x0 = x0_nominal + jr.uniform(key, (2,), minval=-1, maxval=1)
    return x0
