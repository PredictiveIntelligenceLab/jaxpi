from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, hessian
from jax.tree_util import tree_map

import optax

from jaxpi import archs
from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn


class NavierStokes(ForwardIVP):
    def __init__(self, config, t_star, coords, u0, v0, w0, nu):
        super().__init__(config)

        self.u0 = u0
        self.v0 = v0
        self.w0 = w0

        self.t_star = t_star
        self.coords = coords

        self.nu = nu

        self.body_force_fn = lambda x, y: 2 * jnp.sin(4 * jnp.pi * y)

        # Predictions over a grid
        self.u_ic_pred_fn = vmap(self.u_net, (None, None, 0, 0))
        self.v_ic_pred_fn = vmap(self.v_net, (None, None, 0, 0))
        self.w_ic_pred_fn = vmap(self.w_net, (None, None, 0, 0))

        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))
        self.w_pred_fn = vmap(vmap(self.w_net, (None, None, 0, 0)), (None, 0, None, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        t = t / self.t_star[-1]
        z = jnp.stack([t, x, y])
        _, outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        return u, v, p

    def u_net(self, params, t, x, y):
        u, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _ = self.neural_net(params, t, x, y)
        return v

    def p_net(self, params, t, x, y):
        _, _, p = self.neural_net(params, t, x, y)
        return p

    def w_net(self, params, t, x, y):
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        w = v_x - u_y
        return w

    def r_net(self, params, t, x, y):
        u, v, p = self.neural_net(params, t, x, y)

        (u_t, u_x, u_y), (v_t, v_x, v_y), (_, p_x, p_y) = jacrev(self.neural_net, argnums=(1, 2, 3))(params, t, x, y)

        u_hessian = hessian(self.u_net, argnums=(2, 3))(params, t, x, y)
        v_hessian = hessian(self.v_net, argnums=(2, 3))(params, t, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        body_force = self.body_force_fn(x, y)

        # PDE residual
        ru = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy) - body_force
        rv = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        rc = u_x + v_y

        return ru, rv, rc

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        ru_pred, rv_pred, rc_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1], batch[:, 2]
        )

        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        rv_pred = rv_pred.reshape(self.num_chunks, -1)
        rc_pred = rc_pred.reshape(self.num_chunks, -1)

        ru_l = jnp.mean(ru_pred**2, axis=1)
        rv_l = jnp.mean(rv_pred**2, axis=1)
        rc_l = jnp.mean(rc_pred**2, axis=1)

        ru_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ ru_l)))
        rv_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rv_l)))
        rc_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rc_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([ru_gamma, rv_gamma, rc_gamma])
        gamma = gamma.min(0)

        return ru_l, rv_l, rc_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):

        # Unpack batch
        ics_batch = batch["ics"]
        res_batch = batch["res"]

        # Initial condition loss
        coords_batch, u_batch, v_batch, w_batch = ics_batch

        # Initial conditions loss
        u_ic_pred = self.u_ic_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])
        v_ic_pred = self.v_ic_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])
        # w_ic_pred = self.w_ic_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])

        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)
        # w_ic_loss = jnp.mean((w_ic_pred - w_batch) ** 2)

        # residual loss
        if self.config.weighting.use_causal == True:
            ru_l, rv_l, rc_l, gamma = self.res_and_w(params, res_batch)
            ru_loss = jnp.mean(gamma * ru_l)
            rv_loss = jnp.mean(gamma * rv_l)
            rc_loss = jnp.mean(gamma * rc_l)

        else:
            ru_pred, rv_pred, rc_pred = self.r_pred_fn(
                params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            ru_loss = jnp.mean(ru_pred**2)
            rv_loss = jnp.mean(rv_pred**2)
            rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }
        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, coords, u_ref, v_ref, w_ref):
        u_pred = self.u_pred_fn(params, t, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, t, coords[:, 0], coords[:, 1])
        w_pred = self.w_pred_fn(params, t, coords[:, 0], coords[:, 1])

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        w_error = jnp.linalg.norm(w_pred - w_ref) / jnp.linalg.norm(w_ref)

        return u_error, v_error, w_error


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, t, coords, u_ref, v_ref, w_ref):
        u_error, v_error, w_error = self.model.compute_l2_error(
            params,
            t, coords,
            u_ref,
            v_ref,
            w_ref,
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
        self.log_dict["w_error"] = w_error

    def __call__(self, state, batch, t, coords, u_ref, v_ref, w_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, t, coords, u_ref, v_ref, w_ref)

        if self.config.weighting.use_causal:
            _, _, _, causal_weight = self.model.res_and_w(state.params, batch['res'])
            self.log_dict["cas_weight"] = causal_weight.min()

        return self.log_dict
