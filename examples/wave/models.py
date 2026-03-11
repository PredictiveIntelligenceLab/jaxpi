from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class Wave(ForwardIVP):
    def __init__(self, config, u0, t_star, x_star, c=1.0):
        super().__init__(config)

        self.u0 = u0
        self.t_star = t_star
        self.x_star = x_star
        self.c = c

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def u_net(self, params, t, x):
        z = jnp.stack([t, x])
        u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, t, x):
        """Residual: u_tt - c^2 * u_xx = 0"""
        u_tt = grad(grad(self.u_net, argnums=1), argnums=1)(params, t, x)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        return u_tt - self.c ** 2 * u_xx

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        """Compute residuals and weights for causal training."""
        t_sorted = batch[:, 0].sort()
        r_pred = vmap(self.r_net, (None, 0, 0))(params, t_sorted, batch[:, 1])
        r_pred = r_pred.reshape(self.num_chunks, -1)
        l = jnp.mean(r_pred ** 2, axis=1)
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial displacement: u(x, 0) = u0(x)
        u_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        ics_loss = jnp.mean((self.u0 - u_pred) ** 2)

        # Initial velocity: u_t(x, 0) = 0
        u_t_pred = vmap(
            grad(self.u_net, argnums=1), (None, None, 0)
        )(params, self.t0, self.x_star)
        ics_vel_loss = jnp.mean(u_t_pred ** 2)

        # Boundary conditions: u(0, t) = u(1, t) = 0
        u_left = vmap(self.u_net, (None, 0, None))(
            params, self.t_star, self.x_star[0]
        )
        u_right = vmap(self.u_net, (None, 0, None))(
            params, self.t_star, self.x_star[-1]
        )
        bcs_loss = jnp.mean(u_left ** 2) + jnp.mean(u_right ** 2)

        # Residual loss
        if self.config.weighting.use_causal:
            l, w = self.res_and_w(params, batch)
            res_loss = jnp.mean(l * w)
        else:
            r_pred = vmap(self.r_net, (None, 0, 0))(
                params, batch[:, 0], batch[:, 1]
            )
            res_loss = jnp.mean(r_pred ** 2)

        loss_dict = {
            "ics": ics_loss,
            "ics_vel": ics_vel_loss,
            "bcs": bcs_loss,
            "res": res_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        ics_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.u_net, params, self.t0, self.x_star
        )

        ics_vel_ntk = vmap(
            ntk_fn,
            (None, None, None, 0),
        )(grad(self.u_net, argnums=1), params, self.t0, self.x_star)

        bcs_left_ntk = vmap(ntk_fn, (None, None, 0, None))(
            self.u_net, params, self.t_star, self.x_star[0]
        )
        bcs_right_ntk = vmap(ntk_fn, (None, None, 0, None))(
            self.u_net, params, self.t_star, self.x_star[-1]
        )
        bcs_ntk = jnp.concatenate([bcs_left_ntk, bcs_right_ntk])

        if self.config.weighting.use_causal:
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )
            res_ntk = res_ntk.reshape(self.num_chunks, -1)
            res_ntk = jnp.mean(res_ntk, axis=1)
            _, causal_weights = self.res_and_w(params, batch)
            res_ntk = res_ntk * causal_weights
        else:
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {
            "ics": ics_ntk,
            "ics_vel": ics_vel_ntk,
            "bcs": bcs_ntk,
            "res": res_ntk,
        }
        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_ref):
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        return error


class WaveEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(
            params, self.model.t_star, self.model.x_star
        )
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, aspect="auto", origin="lower", cmap="jet")
        plt.colorbar()
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref):
        self.log_dict = {}
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
