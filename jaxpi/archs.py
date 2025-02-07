# Basic Library Imports
from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict

from flax import linen as nn
from flax.core.frozen_dict import freeze

import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant


PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]

identity = lambda x : x

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
                (x.shape[-1], self.features),
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y


# TODO: Make it more general, e.g. imposing periodicity for the given axis


class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x


class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x


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
    




################
#### ActNet ####
################

# from https://www.wolframalpha.com/input?i=E%5B%28sin%28wx%2Bp%29%29%5D+where+x+is+normally+distributed
def _mean_transf(mu, sigma, w, p):
    """ Mean of the R.V. Y=sin(w*X+p) when X is normally distributed with mean mu
    and standard deviation sigma.

    Args:
        mu: mean of the R.V. X.
        sigma: standard deviation of the R.V. X.
        w: frequency of the sinusoidal transformation.
        p: phase of the sinusoidal transformation.

    Returns:
        The mean of the transformed R.V. Y.
    """
    return jnp.exp(-0.5* (sigma*w)**2) * jnp.sin(p + mu*w)

# from https://www.wolframalpha.com/input?i=E%5Bsin%28wx%2Bp%29%5E2%5D+where+x+is+normally+distributed
def _var_transf(mu, sigma, w, p):
    """ Variance of the R.V. Y=sin(w*X+p) when X is normally distributed with
    mean mu and standard deviation sigma.

    Args:
        mu: mean of the R.V. X.
        sigma: standard deviation of the R.V. X.
        w: frequency of the sinusoidal transformation.
        p: phase of the sinusoidal transformation.

    Returns:
        The variance of the transformed R.V. Y.
    """
    return 0.5 - 0.5*jnp.exp(-2 * ((sigma*w)**2))*jnp.cos(2*(p+mu*w)) \
        -_mean_transf(mu, sigma, w, p)**2

class ActLayer(nn.Module):
    """A ActLayer module. 
    
    For further details on standard choices of initializers, please refer to
    Appendix D of the paper: https://arxiv.org/pdf/2410.01990

    Attributes:
        out_dim: output dimension of ActLayer.
        num_freqs: number of frequencies/basis functions of the ActLayer.
        use_bias: whether to add bias the the output (default: True).
        freqs_init: initializer for basis function frequencies.
        phases_init: initializer for basis function phases.
        beta_init: initializer for beta parameter.
        lamb_init: initializer for lambda parameter.
        bias_init: initializer for bias parameter.
        freze_basis: whether to freeze gradients passing thorough basis
            functions (default: False).
        freq_scaling: whether to scale basis functions to ensure mean 0 and
            standard deviation 1 (default: True).
        freq_scaling_eps: small epsilon added to the denominator of frequency
            scaling for numerical stability (default: 1e-3).
        precision: numerical precision of the computation. See
        ``jax.lax.Precision`` for details. (default: None)
    """
    out_dim : int
    num_freqs : int
    use_bias : bool=True
    # parameter initializers
    freqs_init : Callable=nn.initializers.normal(stddev=1.)  # normal entries w/ mean zero
    phases_init : Callable=nn.initializers.zeros
    beta_init : Callable=nn.initializers.variance_scaling(1., 'fan_in', distribution='uniform')
    lamb_init : Callable=nn.initializers.variance_scaling(1., 'fan_in', distribution='uniform')
    bias_init : Callable=nn.initializers.zeros
    # other configurations
    freeze_basis : bool=False
    softmax_lamb : bool=False
    freq_scaling : bool=True
    freq_scaling_eps : float=1e-3 # used for numerical stability of gradients
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        """Forward pass of an ActLayer.

        Args:
            x: The nd-array to be transformed.

        Returns:
            The transformed input x.
        """
        # x should initially be shape (d,)

        # initialize trainable parameters
        freqs = self.param('freqs',
                           self.freqs_init,
                           (1,self.num_freqs)) # shape (1, num_freqs)
        phases = self.param('phases',
                            self.phases_init,
                            (1,self.num_freqs)) # shape (1, num_freqs)
        beta = self.param('beta',
                          self.beta_init,
                          (self.num_freqs, self.out_dim)) # shape (num_freqs, out_dim)
        lamb = self.param('lamb',
                          self.lamb_init,
                          (x.shape[-1], self.out_dim)) # shape (d, out_dim)
        if self.softmax_lamb:
          lamb = nn.softmax(lamb, axis=0)

        if self.freeze_basis:
            freqs = jax.lax.stop_gradient(freqs)
            phases = jax.lax.stop_gradient(phases)
        
        # perform basis expansion
        x = jnp.expand_dims(x, 1) # shape (d, 1)
        x = jnp.sin(freqs*x + phases) # shape (d, num_freqs)
        if self.freq_scaling:
            x = (x - _mean_transf(0., 1., freqs, phases)) \
                / (jnp.sqrt(self.freq_scaling_eps + _var_transf(0., 1., freqs, phases)))

        # efficiently computes eq 6 from https://arxiv.org/pdf/2410.01990 using
        # einsum. Depending on hardware and JAX/CUDA version, there may be
        # slightly faster alternatives, but we chose this one for the sake of
        # simplicity/clarity.
        x = jnp.einsum('ij,jk,ik -> k', x, beta, lamb, precision=self.precision)

        # optionally add bias
        if self.use_bias:
           bias = self.param('bias',
                             self.bias_init,
                             (self.out_dim,))
           x = x + bias # Shape (out_dim,)

        return x # Shape out_dim,)
    

class ActNet(nn.Module):
    """A ActNet module.

    Attributes:
        embed_dim: embedding dimension for ActLayers.
        num_layers: how many intermediate blocks are used.
        out_dim: output dimension of ActNet.
        num_freqs: number of frequencies/basis functions of the ActLayers.
        output_activation: output_activation: activation for last layer of
            network (default: identity).
        op_order: order of operations contained in each intermediate block. This
            should be a string containing only 'A' (ActLayer), 'S' (Skip
            connection) or 'L' (LayerNorm) characters. (default: 'A')
        use_act_bias: whether to add bias the the output of ActLayers
            (default: True).
        freqs_init: initializer for basis function frequencies of ActLayers.
        phases_init: initializer for basis function phases of ActLayers.
        beta_init: initializer for beta parameter of ActLayers.
        lamb_init: initializer for lambda parameter of ActLayers.
        act_bias_init: initializer for bias parameter of ActLayers.
        proj_bias_init: initializer for bias parameter of initial projection
            Layer.
        w0_init: initializer for w0 scale parameter.
        w0_fixed: if False, initializes w0 using w0_init. Otherwise uses given
            fixed w0 (default: False).
        freze_basis: whether to freeze gradients passing thorough basis
            functions (default: False).
        freq_scaling: whether to scale basis functions to ensure mean 0 and
            standard deviation 1 (default: True).
        freq_scaling_eps: small epsilon added to the denominator of frequency
            scaling for numerical stability (default: 1e-3).
        precision: numerical precision of the computation. See
        ``jax.lax.Precision`` for details. (default: None)
    """
    embed_dim : int
    num_layers : int
    out_dim : int
    num_freqs : int
    arch_name: Optional[str] = "ActNet"
    output_activation : Callable = identity
    periodicity: Union[None, Dict] = None
    op_order : str='A'
    # op_order should be a string containing only 'A' (ActLayer), 'S' (Skip
    # connection) or 'L' (LayerNorm) characters. This feature was used for
    # development/debugging, but is not used in any experiment of the paper.

    # parameter initializers
    freqs_init : Callable=nn.initializers.normal(stddev=1.)  # normal entries w/ mean zero
    phases_init : Callable=nn.initializers.zeros
    beta_init : Callable=nn.initializers.variance_scaling(1., 'fan_in', distribution='uniform')
    lamb_init : Callable=nn.initializers.variance_scaling(1., 'fan_in', distribution='uniform')
    act_bias_init : Callable=nn.initializers.zeros
    proj_bias_init : Callable=lambda key, shape, dtype : random.uniform(key, shape, dtype, minval=-jnp.sqrt(3), maxval=jnp.sqrt(3))
    
    w0_init : Callable=nn.initializers.constant(30.) # following SIREN strategy
    w0_fixed : float=5.0

    # other ActLayer configurations
    use_act_bias : bool=True
    freeze_basis : bool=False
    softmax_lamb : bool=False
    freq_scaling : bool=True
    freq_scaling_eps : float=1e-3 # used for numerical stability of gradients
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        """Forward pass of an ActNet.

        Args:
            x: The nd-array to be transformed.

        Returns:
            The transformed input x.
        """
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        # initialize w0 parameter
        if self.w0_fixed is False:
            # trainable scalar parameter
            w0 = self.param('w0',
                            self.w0_init,
                            ())
            # use softplus to ensure w0 is positive and does not decay to zero too fast
            w0 = nn.softplus(w0)
        else: # use user-specified value for w0
            w0 = self.w0_fixed
        # project to embeded dimension
        x = x*w0
        x = nn.Dense(self.embed_dim, bias_init=self.proj_bias_init, precision=self.precision)(x)
        
        for _ in range(self.num_layers):
            y = x # store initial value as x, do operations on y
            for char in self.op_order:
                if char == 'A': # ActLayer
                    y  = ActLayer(
                            out_dim = self.embed_dim,
                            num_freqs = self.num_freqs,
                            use_bias = self.use_act_bias,
                            freqs_init = self.freqs_init,
                            phases_init = self.phases_init,
                            beta_init = self.beta_init,
                            lamb_init = self.lamb_init,
                            bias_init = self.act_bias_init,
                            freeze_basis = self.freeze_basis,
                            softmax_lamb = self.softmax_lamb,
                            freq_scaling = self.freq_scaling,
                            freq_scaling_eps = self.freq_scaling_eps,
                            precision=self.precision,
                            )(y)
                elif char == 'S': # Skip connection
                    y = y + x
                elif char == 'L': # LayerNorm
                    y = nn.LayerNorm()(y)
                else:
                    raise NotImplementedError(f"Could not recognize option '{char}'. Options for op_order should be 'A' (ActLayer), 'S' (Skip connection) or 'L' (LayerNorm).")
            x = y # update value of x after all operations are done

        # project to output dimension and potentially use output activation
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.he_uniform(), precision=self.precision)(x)
        x = self.output_activation(x)

        return x
