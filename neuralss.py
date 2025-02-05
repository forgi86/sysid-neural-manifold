import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as jr


def f(params, scalers, x, u):
    x_lin = params["f"]["lin"]["A"] @ x + params["f"]["lin"]["B"] @ u
    zx_nl = nn.gelu(
        params["f"]["nn"]["Wfx"] @ x
        + params["f"]["nn"]["Wfu"] @ u
        + params["f"]["nn"]["bf"]
    )
    x_nl = params["f"]["nn"]["Wx"] @ zx_nl + params["f"]["nn"]["bx"]
    x_new = scalers["f"]["lin"] * x_lin + scalers["f"]["nl"] * x_nl
    return x_new


def g(params, scalers, x, u):
    y_lin = params["g"]["lin"]["C"] @ x  # + params["g"]["lin"]["D"] @ u
    zy_nl = nn.gelu(
        params["g"]["nn"]["Wgx"] @ x
        # + params["g"]["nn"]["Wgu"] @ u
        + params["g"]["nn"]["bg"]
    )
    y_nl = params["g"]["nn"]["Wy"] @ zy_nl + params["g"]["nn"]["by"]
    y = scalers["g"]["lin"] * y_lin + scalers["g"]["nl"] * y_nl
    return y


def fg(params, scalers, x, u):
    x_new = f(params, scalers, x, u)
    y = g(params, scalers, x, u)
    return x_new, y


def ss_apply(params, scalers, x, u):
    fg_cfg = lambda x, u: fg(params, scalers, x, u)
    _, y = jax.lax.scan(fg_cfg, x, u)
    return y


def ss_state_apply(params, scalers, x, u):
    def fg_cfg(
        x,
        u,
    ):
        x_new = f(params, scalers, x, u)
        return x_new, x

    _, y = jax.lax.scan(fg_cfg, x, u)
    return y


def ss_init(key, nu=1, ny=1, nx=3, hidden_f=16, hidden_g=16):

    params = {}
    params["f"] = {}
    params["f"]["nn"] = {}
    params["f"]["lin"] = {}
    params["g"] = {}
    params["g"]["nn"] = {}
    params["g"]["lin"] = {}

    # f nn
    key, subkey = jr.split(key)
    params["f"]["nn"]["Wfx"] = jr.normal(subkey, shape=(hidden_f, nx)) / jnp.sqrt(nx)
    key, subkey = jr.split(key)
    params["f"]["nn"]["Wfu"] = jr.normal(subkey, shape=(hidden_f, nu)) / jnp.sqrt(nu)
    key, subkey = jr.split(key)
    params["f"]["nn"]["bf"] = jnp.zeros(shape=(hidden_f,))
    
    key, subkey = jr.split(key)
    params["f"]["nn"]["Wx"] = jr.normal(subkey, shape=(nx, hidden_f)) / jnp.sqrt(hidden_f)
    key, subkey = jr.split(key)
    params["f"]["nn"]["bx"] = jnp.zeros(shape=(nx,))

    # f lin
    key, subkey = jr.split(key)
    params["f"]["lin"]["A"] = jr.normal(subkey, shape=(nx, nx))
    key, subkey = jr.split(key)
    params["f"]["lin"]["B"] = jr.normal(subkey, shape=(nx, nu))

    # g nn
    key, subkey = jr.split(key)
    params["g"]["nn"]["Wgx"] = jr.normal(subkey, shape=(hidden_g, nx)) / jnp.sqrt(nx)
    key, subkey = jr.split(key)
    params["g"]["nn"]["Wgu"] = jr.normal(subkey, shape=(hidden_g, nu)) / jnp.sqrt(nu)
    key, subkey = jr.split(key)
    params["g"]["nn"]["bg"] = jnp.zeros((hidden_g,))
    key, subkey = jr.split(key)
    params["g"]["nn"]["Wy"] = jr.normal(subkey, shape=(ny, hidden_g)) / jnp.sqrt(hidden_g)
    key, subkey = jr.split(key)
    params["g"]["nn"]["by"] = jnp.zeros((ny,))

    # g lin
    key, subkey = jr.split(key)
    params["g"]["lin"]["C"] = jr.normal(subkey, shape=(ny, nx))
    key, subkey = jr.split(key)
    params["g"]["lin"]["D"] = jr.normal(subkey, shape=(ny, nu))

    return params