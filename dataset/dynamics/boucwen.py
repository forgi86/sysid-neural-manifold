import jax.numpy as jnp
import jax.random as jr

# params from wikipedia
params_nominal = jnp.array([2.0, 10.0, 5e4, 5e4, 1e3, 0.8, -1.1, 1.0]) 
params_min = jnp.array([1.0, 5.0, 2.5e4, 2.5e4, .5e3, .5, -1.5, 1.0]) 
params_max = jnp.array([3.0, 15.0, 7.5e4, 7.5e4, 4.5e3, 0.9, -0.5, 1.0]) 


x0_nominal = jnp.array([0.0, 0.0, 0.0]) 

def f_xu(x, u, params):
    """ Duffing oscillator"""
    p, v, z = x # position, velocity
    m, c, k, alpha, beta, gamma, delta, nu = params

    
    F = u[0] # F, = u  

    r = k*p + c*v # restoration force

    dp = v
    dv = (F - r - z)/m
    dz = alpha * v - beta*(gamma * jnp.abs(v) * (jnp.abs(z)**(nu - 1)) * z + delta * v * (jnp.abs(z)**nu) ) # histeretic dynamics

    dx = jnp.array([dp, dv, dz])
    return dx

# currenty unused
# def g_x(x):
#     return x

def params_fn(key):
    #params_nominal = jnp.array([-1.0, 0.25, 0.1, 2.5, 2]) 
    #params = params_nominal * jr.uniform(key, params_nominal.shape, minval=0.9, maxval=1.1)
    params = jr.uniform(key, shape=params_min.shape, minval=params_min, maxval=params_max)
    return params

def init_fn(key):
    x0 = x0_nominal #+ jr.normal(key, (3,)) * jnp.array([2.5e-5, 5.8e-3, 1.3])
    return x0

def init_fn_randn(key):
    x0 = x0_nominal + jr.normal(key, (3,)) * jnp.array([3.1e-05, 7.2e-03, 1.6e+00])
    return x0

# def simulate_newmark(x0, t, u, params, f_xu):
#     ts = t[1] - t[0]
#     fn_rk = discretize_rk4(f_xu, ts)
#     _, x_sim = jax.lax.scan(lambda x, u: fn_rk(x, u, params), x0, u)
#     return x_sim