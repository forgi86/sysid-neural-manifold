from jaxid.common import MLP
import jax.numpy as jnp
from typing import Callable, List
import flax.linen as nn


class Encoder(nn.Module):
    mlp_layers: List[int]
    rnn_size: int = 128

    def setup(self):
        self.rnn = nn.Bidirectional(
            nn.RNN(nn.GRUCell(self.rnn_size)), nn.RNN(nn.GRUCell(self.rnn_size))
        )
        self.mlp = MLP(self.mlp_layers)

    def __call__(self, y, u):
        yu = jnp.concat((y, u), axis=-1)
        rnn_feat = self.rnn(yu).mean(axis=-2)
        z = self.mlp(rnn_feat)
        return z
    

class Projector(nn.Module):
    outputs: int
    unflatten: Callable

    def setup(self):
        self.net = nn.Dense(self.outputs, use_bias=False)


    def __call__(self, z):
        out = self.net(z)
        return self.unflatten(out)


class EncoderProjector(nn.Module):
    outputs: int
    unflatten: Callable
    mlp_layers: List[int]
    rnn_size: int = 128

    def setup(self):
        self.rnn = nn.Bidirectional(
            nn.RNN(nn.GRUCell(self.rnn_size)), nn.RNN(nn.GRUCell(self.rnn_size))
        )
        self.mlp = MLP(self.mlp_layers)
        self.out_proj = nn.Dense(self.outputs, use_bias=False)


    def __call__(self, y, u):
        yu = jnp.concat((y, u), axis=-1)
        rnn_feat = self.rnn(yu).mean(axis=-2)
        mlp_feat = self.mlp(rnn_feat)
        out = self.out_proj(mlp_feat)
        return self.unflatten(out)
