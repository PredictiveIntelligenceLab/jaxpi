from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict


from flax import linen as nn
from flax.core.frozen_dict import freeze

import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v

    return init


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied
    trainable: Tuple[
        bool
    ]  # Specifies whether the period for each axis is trainable or not

    def setup(self):
        # Initialize period parameters as trainable or constant and store them in a flax frozen dict
        period_params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                period_params[f"period_{idx}"] = self.param(
                    f"period_{idx}", constant(self.period[idx]), ()
                )
            else:
                period_params[f"period_{idx}"] = self.period[idx]

        self.period_params = freeze(period_params)

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []

        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Embedding(nn.Module):
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        return x


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )

        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features)
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y


class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class Bottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.output_dim, reparam=self.reparam)(x)

        x = (
            x + identity
        )  # Please note that the skip connection is added before the activation function, which is the same as the original ResNet

        x = self.activation_fn(x)

        return x


class PIBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        """
        Physics-informed bottleneck block: Add the skip connection after the activation function,
        which is different from the original ResNet, making it an identity mapping at initialization
        """
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = Dense(features=self.output_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        # alpha = jnp.exp(-alpha)

        x = alpha * x + (1 - alpha) * identity

        return x


class PIModifiedBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x, u, v):
        identity = x

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = Dense(features=self.output_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)

        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        x = alpha * x + (1 - alpha) * identity

        return x


class ResNet(nn.Module):
    arch_name: Optional[str] = "ResNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Bottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                reparam=self.reparam,
            )(x)

        y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class PIResNet(nn.Module):
    arch_name: Optional[str] = "PIResNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = PIBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam,
            )(x)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class PirateNet(nn.Module):
    arch_name: Optional[str] = "PirateNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        embs = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)
        x = embs

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        u = self.activation_fn(u)

        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam
            )(x, u, v)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


#################################################################################################
#################################### neural operators ###########################################
#################################################################################################

class MlpBlock(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    activation: str
    reparam: Union[None, Dict]
    final_activation: bool

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        if self.final_activation:
            x = self.activation_fn(x)

        return x


class DeepONet(nn.Module):
    arch_name: Optional[str] = "DeepONet"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u, x):
        u = MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            final_activation=False,
            reparam=self.reparam,
        )(u)

        x = Mlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)

        y = u * x
        y = self.activation_fn(y)
        y = Dense(features=self.out_dim, reparam=self.reparam)(y)
        return y
