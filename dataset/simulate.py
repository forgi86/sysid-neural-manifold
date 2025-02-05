import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import diffrax
from .input.signals import multisine_signal
from .input.interpolation import ZOHInterpolation as Interpolation
#from diffrax import LinearInterpolation as Interpolation


def simulate_diffrax(x0, t, u, params, f_xu):

    u_interp = Interpolation(ts=t, ys=u)
    def vector_field(t, x, args):
        ut = u_interp.evaluate(t)
        dx = f_xu(x, ut, args)
        return dx
    
    dt0 = (t[1] - t[0])
    #dt0=1e-3
    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vector_field),
        #solver=diffrax.Dopri5(),
        solver=diffrax.Tsit5(),
        t0=t[0],
        t1=t[-1],
        dt0=dt0,
        y0=x0,
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6, jump_ts=t[:-1]), # last steps seems lead to infinities in the result
        args=params,
        max_steps=None
    )

    xs = sol.ys
    #ys = g_x(xs)

    return xs


def generate_batch(key, systems, runs, init_fn, input_fn, params_fn, simulate_fn):

    init_key, input_key, params_key = jr.split(key, 3)

    input_keys = jr.split(input_key, systems * runs)
    batch_t, batch_u = jax.vmap(input_fn)(input_keys)
    batch_u = batch_u.reshape((systems, runs) + batch_u.shape[1:]) # batch_size, K, seq_len, 1
    batch_t = batch_t.reshape((systems, runs) + batch_t.shape[1:]) # batch_size, K, seq_len


    init_keys = jr.split(init_key, systems * runs)
    batch_x0 = jax.vmap(init_fn)(init_keys)
    batch_x0 = batch_x0.reshape((systems, runs) + batch_x0.shape[1:]) # batch_size, K, nx

    params_keys = jr.split(params_key, systems)
    params = jax.vmap(params_fn)(params_keys) # batch_size, num_params

    simulate_reps = jax.vmap(simulate_fn, in_axes=(0, 0, 0, None)) # solve K repetitions for one system
    simulate_batch = jax.vmap(simulate_reps, in_axes=(0, 0, 0, 0)) # repeat for all systems
    batch_x = simulate_batch(batch_x0, batch_t, batch_u, params)

    # possible small optimization: ts = batch_t[0, 0] and do not vmap the time axis
    #ts = batch_t[0, 0]
    #simulate_reps = jax.vmap(simulate_fn, in_axes=(0, None, 0, None)) # solve K repetitions for one system
    #simulate_batch = jax.vmap(simulate_reps, in_axes=(0, None, 0, 0)) # repeat for all systems
    #batch_x = simulate_batch(batch_x0, ts, batch_u, params)

    return batch_u, batch_x, batch_t, params


def simulate_euler(x0, t, u, params, f_xu):
    ts = t[1] - t[0]
    fn_rk = discretize_euler(f_xu, ts)
    _, x_sim = jax.lax.scan(lambda x, u: fn_rk(x, u, params), x0, u)
    return x_sim


def simulate_rk4(x0, t, u, params, f_xu):
    ts = t[1] - t[0]
    fn_rk = discretize_rk4(f_xu, ts)
    _, x_sim = jax.lax.scan(lambda x, u: fn_rk(x, u, params), x0, u)
    return x_sim


def discretize_rk4(fun_ct, dt):
    def fun_rk(x, u, args):
        dt2 = dt/2
        k1 = fun_ct(x, u, args)
        k2 = fun_ct(x + dt2 * k1, u, args)
        k3 = fun_ct(x + dt2 * k2, u, args)
        k4 = fun_ct(x + dt * k3, u, args)
        dx = dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_new = x + dx
        return x_new, x
    return fun_rk

def discretize_euler(fun_ct, dt):
    def fun_euler(x, u, args):
        k1 = fun_ct(x, u, args)
        dx = dt * k1
        x_new = x + dx
        return x_new, x
    return fun_euler