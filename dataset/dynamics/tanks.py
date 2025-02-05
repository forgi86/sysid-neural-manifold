import jax.numpy as jnp
import jax.random as jr

# params from masti et al
params_nominal = jnp.array([0.5, 0.4, 0.2, 0.3])
u_eq = 4.0

def x_eq(params, u):
    """ equilibrium x, given u"""
    k1, k2, k3, k4 = params
    x1_eq = (k2 * u / k1)**2
    x2_eq = (k3/k4)**2 * x1_eq
    return jnp.array([x1_eq, x2_eq])

#x0_nominal = jnp.array([11.0, 7.0]) 
x0_nominal = x_eq(params_nominal, u_eq)

def f_xu(x, u, params):
    """ tanks dynamics """
    x1, x2 = x # position, velocity
    k1, k2, k3, k4 = params
    u = u[0] + u_eq
    dx1 = -k1 * jnp.sqrt(x1)  + k2*u
    dx2 =  k3 * jnp.sqrt(x1)  -k4*jnp.sqrt(x2)
    dx = jnp.array([dx1, dx2])
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

def params_init_fn(key):
    params_key, init_key = key.split(2)
    params = params_fn(params_key)
    x0 = x_eq + jr.uniform(init_key, (2,), minval=-1, maxval=1)
    return params, x0
