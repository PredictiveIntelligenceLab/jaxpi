from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian
from jax.flatten_util import ravel_pytree

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn

from utils import sample_points_on_square_boundary

from matplotlib import pyplot as plt


class NavierStokes2D(ForwardBVP):
    def __init__(self, config):
        super().__init__(config)

        # Sample boundary points uniformly
        num_pts = 256
        self.x_bc1 = sample_points_on_square_boundary(
            num_pts, eps=0.01
        )  # avoid singularity a right corner for u velocity
        self.x_bc2 = sample_points_on_square_boundary(num_pts, eps=0.01)

        # Boundary conditions
        self.v_bc = jnp.zeros((num_pts * 4,))
        self.u_bc = self.v_bc.at[:num_pts].set(1.0)

        # Predictions over a grid
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, None, 0, 0))

    def neural_net(self, params, x, y):
        z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        return u, v, p

    def u_net(self, params, x, y):
        u, _, _ = self.neural_net(params, x, y)
        return u

    def v_net(self, params, x, y):
        _, v, _ = self.neural_net(params, x, y)
        return v

    def p_net(self, params, x, y):
        _, _, p = self.neural_net(params, x, y)
        return p

    def r_net(self, params, nu, x, y):
        u, v, p = self.neural_net(params, x, y)

        (u_x, u_y), (v_x, v_y), (p_x, p_y) = jacrev(self.neural_net, argnums=(1, 2))(params, x, y)

        u_hessian = hessian(self.u_net, argnums=(1, 2))(params, x, y)
        v_hessian = hessian(self.v_net, argnums=(1, 2))(params, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        ru = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        rv = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        rc = u_x + v_y

        return ru, rv, rc

    def ru_net(self, params, nu, x, y):
        ru, _, _ = self.r_net(params, nu, x, y)
        return ru

    def rv_net(self, params, nu, x, y):
        _, rv, _ = self.r_net(params, nu, x, y)
        return rv

    def rc_net(self, params, nu, x, y):
        _, _, rc = self.r_net(params, nu, x, y)
        return rc

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, nu):
        # boundary condition losses
        # Compute forward pass of u and v
        u_pred = self.u_pred_fn(params, self.x_bc1[:, 0], self.x_bc1[:, 1])
        v_pred = self.v_pred_fn(params, self.x_bc2[:, 0], self.x_bc2[:, 1])

        # Compute losses
        u_bc_loss = jnp.mean((u_pred - self.u_bc) ** 2)
        v_bc_loss = jnp.mean(v_pred**2)

        # Compute forward pass of residual
        ru_pred, rv_pred, rc_pred = self.r_pred_fn(params, nu, batch[:, 0], batch[:, 1])
        # Compute losses
        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            "u_bc": u_bc_loss,
            "v_bc": v_bc_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch, nu):
        u_bc_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_net, params, self.x_bc1[:, 0], self.x_bc1[:, 1]
        )
        v_bc_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.v_net, params, self.x_bc2[:, 0], self.x_bc2[:, 1]
        )

        ru_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.ru_net, params, nu, batch[:, 0], batch[:, 1]
        )
        rv_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.rv_net, params, nu, batch[:, 0], batch[:, 1]
        )
        rc_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.rc_net, params, nu, batch[:, 0], batch[:, 1]
        )

        ntk_dict = {
            "u_bc": u_bc_ntk,
            "v_bc": v_bc_ntk,
            "ru": ru_ntk,
            "rv": rv_ntk,
            "rc": rc_ntk,
        }

        return ntk_dict

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0, 3))
    def update_weights(self, state, batch, nu):
        weights = self.compute_weights(state.params, batch, nu)
        weights = lax.pmean(weights, "batch")
        state = state.apply_weights(weights=weights)
        return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0, 3))
    def step(self, state, batch, nu):
        grads = grad(self.loss)(state.params, state.weights, batch, nu)
        grads = lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        return state

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, x_star, y_star, U_test):
        u_pred = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        v_pred = vmap(vmap(self.v_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )

        U_pred = jnp.sqrt(u_pred**2 + v_pred**2)
        l2_error = jnp.linalg.norm(U_pred - U_test) / jnp.linalg.norm(U_test)

        return l2_error


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, x_star, y_star, U_ref):
        l2_error = self.model.compute_l2_error(params, x_star, y_star, U_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params, x_star, y_star):
        u_pred = vmap(vmap(self.model.u_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        v_pred = vmap(vmap(self.model.v_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        U_pred = jnp.sqrt(u_pred**2 + v_pred**2)

        fig = plt.figure()
        plt.pcolor(U_pred.T, cmap="jet")
        self.log_dict["U_pred"] = fig
        fig.close()

    def __call__(self, state, batch, x_star, y_star, U_ref, nu):
        self.log_dict = super().__call__(state, batch, nu)

        if self.config.logging.log_errors:
            self.log_errors(state.params, x_star, y_star, U_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params, x_star, y_star)

        return self.log_dict
