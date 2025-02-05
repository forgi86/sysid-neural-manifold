import jax.numpy as jnp
import jax.random as jr


def multisine(key, N, pmin=1, pmax=21, P=1):
    uf = jnp.zeros((N//2 + 1,), dtype=complex)
    lines = pmax - pmin + 1
    components =  jnp.exp(1j*jr.uniform(key, minval=0, maxval=jnp.pi*2, shape=(lines,)))
    uf = uf.at[pmin:pmax+1].set(components)
#    for p in range(pmin, pmax):
#        key, subkey = jr.split(key)
#        uf = uf.at[p].set(jnp.exp(1j*jr.uniform(subkey, minval=0, maxval=jnp.pi*2)))

    us = jnp.fft.irfft(uf/2)
    us /= jnp.std(us)
    us = jnp.concatenate([us] * P).reshape(-1, 1)
    return us


def multisine_signal(key, seq_len, fs=1.0, fh=0.1, scale=1.0):
    ts = 1/fs
    t = jnp.arange(seq_len) * ts
    pmax = int(seq_len*fh/fs) + 1 # maximum frequency
    u = scale*multisine(key, N=seq_len, pmax=pmax)
    return t, u


def zero_signal(key, seq_len, fs=1.0):
    ts = 1/fs
    t = jnp.arange(seq_len) * ts
    u = jnp.zeros((seq_len, 1))
    return t, u

