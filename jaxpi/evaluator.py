import jax.numpy as jnp

from jax import jacrev
from jax.tree_util import tree_map

from jaxpi.utils import flatten_pytree


class BaseEvaluator:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.log_dict = {}

    def log_losses(self, params, batch, *args):
        losses = self.model.losses(params, batch, *args)

        for key, values in losses.items():
            self.log_dict[key + "_loss"] = values

    def log_weights(self, state):
        weights = state.weights
        for key, values in weights.items():
            self.log_dict[key + "_weight"] = values

    def log_grads(self, params, batch, *args):
        grads = jacrev(self.model.losses)(params, batch, *args)
        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            grad_norm = jnp.linalg.norm(flattened_grad)
            self.log_dict[key + "_grad_norm"] = grad_norm

    def log_ntk(self, params, batch, *args):
        ntk = self.model.compute_diag_ntk(params, batch)
        mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)

        for key, values in mean_ntk_dict.items():
            self.log_dict[key + "_ntk"] = values

    def __call__(self, state, batch, *args):
        # Initialize the log dict
        self.log_dict = {}
        params = state.params

        if self.config.logging.log_losses:
            self.log_losses(params, batch, *args)

        if self.config.logging.log_weights:
            self.log_weights(state)

        if self.config.logging.log_grads:
            self.log_grads(params, batch, *args)

        if self.config.logging.log_ntk:
            self.log_ntk(params, batch, *args)

        return self.log_dict
