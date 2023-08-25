from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class PINN(ForwardBVP):

    def __init__(self, config, x_star):
        super().__init__(config)

        self.x_star = x_star
        self.x_bc = jnp.array([x_star[0], x_star[-1]])

        # forcing term
        self.forcing = lambda x: (jnp.pi ** 2 * jnp.cos(jnp.pi * x) + jnp.cos(jnp.pi * x) ** 3)

        # Predictions over a grid
        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.r_pred_fn = (self.r_net, (None, 0))

    def u_net(self, params, x):
        z = jnp.stack([x])
        u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, x):
        u = self.u_net(params, x)
        u_xx = grad(grad(self.u_net, argnums=1), argnums=1)(params, x)
        return - u_xx + u**3 - self.forcing(x)**2


    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        u_pred = vmap(self.u_net, (None, 0))(params, self.x_bc)
        bcs_loss = jnp.mean(u_pred ** 2)

        # Residual loss
        r_pred = self.r_pred_fn(params, self.x_star)
        res_loss = jnp.mean(r_pred ** 2)

        loss_dict = {"bcs": bcs_loss, "res": res_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        bcs_ntk = vmap(ntk_fn, (None, None, 0))(
            self.u_net, params, self.x_bc
        )

        res_ntk = vmap(ntk_fn, (None, 0))(
            self.r_net, params, self.x_star
        )

        ntk_dict = {"bcs": bcs_ntk, "res": res_ntk}

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error






# class NaturalGrad(ForwardBVP):



class PINNEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        return self.log_dict
