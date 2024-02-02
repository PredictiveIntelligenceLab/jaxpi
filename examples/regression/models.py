from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import PINN
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class Regression(PINN):
    def __init__(self, config, x_star, u_star):
        super().__init__(config)

        self.x_star = x_star
        self.u_star = u_star

        self.derivatives = {
            0: self.u_net,
            1: self.u_x_net,
            2: self.u_xx_net,
            3: self.u_xxx_net,
            4: self.u_xxxx_net,
        }

        self.pred_fn = self.derivatives[config.deriv_order]

    def u_net(self, params, x):
        z = jnp.stack([x])
        _, u = self.state.apply_fn(params, z)
        return u[0]

    def u_x_net(self, params, x):
        u_x = grad(self.u_net, argnums=1)(params, x)
        return u_x

    def u_xx_net(self, params, x):
        u_xx = grad(grad(self.u_net, argnums=1), argnums=1)(params, x)
        return u_xx

    def u_xxx_net(self, params, x):
        u_xxx = grad(grad(grad(self.u_net, argnums=1), argnums=1), argnums=1)(params, x)
        return u_xxx

    def u_xxxx_net(self, params, x):
        u_xxxx = grad(
            grad(grad(grad(self.u_net, argnums=1), argnums=1), argnums=1), argnums=1
        )(params, x)
        return u_xxxx

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        x_star, u_star = batch

        pred = vmap(self.pred_fn, (None, 0))(params, x_star)
        loss = jnp.mean((pred - u_star) ** 2)

        loss_dict = {"mse": loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params):
        pred = vmap(self.pred_fn, (None, 0))(params, self.x_star)
        error = jnp.linalg.norm(pred - self.u_star) / jnp.linalg.norm(self.u_star)
        return error


class RegressionEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params):
        l2_error = self.model.compute_l2_error(params)
        self.log_dict["l2_error"] = l2_error

    def __call__(self, state, batch):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
