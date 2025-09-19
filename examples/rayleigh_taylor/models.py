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
    def __init__(self, config, alpha1, alpha2, alpha3, alpha4, t_star, coords, u0, v0, p0, temp0, velocity_scale=1.0):
        super().__init__(config)

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

        self.u0 = u0
        self.v0 = v0
        self.p0 = p0
        self.temp0 = temp0

        self.t_star = t_star
        self.coords = coords

        velocity_scale = jnp.max(jnp.sqrt(u0 ** 2 + v0 ** 2))
        self.velocity_scale = velocity_scale + 1e-3

        # Predictions over a grid
        self.u_ic_pred_fn = vmap(self.u_net, (None, None, 0, 0))
        self.v_ic_pred_fn = vmap(self.v_net, (None, None, 0, 0))
        self.p_ic_pred_fn = vmap(self.p_net, (None, None, 0, 0))
        self.temp_ic_pred_fn = vmap(self.temp_net, (None, None, 0, 0))

        self.u_bc_pred_fn = vmap(self.u_net, (None, 0, 0, 0))
        self.v_bc_pred_fn = vmap(self.v_net, (None, 0, 0, 0))
        self.temp_bc_pred_fn = vmap(self.temp_net, (None, 0, 0, 0))

        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))
        self.p_pred_fn = vmap(vmap(self.p_net, (None, None, 0, 0)), (None, 0, None, None))
        self.temp_pred_fn = vmap(vmap(self.temp_net, (None, None, 0, 0)), (None, 0, None, None))

        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        t = t / self.t_star[-1]
        z = jnp.stack([t, x, y])
        _, outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        temp = outputs[3]

        # scale = jnp.exp(outputs[4]) + self.velocity_scale
        # u = u * scale
        # v = v * scale

        return u, v, p, temp


    def u_net(self, params, t, x, y):
        u, _, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _, _ = self.neural_net(params, t, x, y)
        return v
    def p_net(self, params, t, x, y):
        _, _, p, _ = self.neural_net(params, t, x, y)
        return p

    def temp_net(self, params, t, x, y):
        _, _, _, temp = self.neural_net(params, t, x, y)
        return temp

    def r_net(self, params, t, x, y):
        u, v, p, temp = self.neural_net(params, t, x, y)

        ((u_t, u_x, u_y),
         (v_t, v_x, v_y),
         (_, p_x, p_y),
         (temp_t, temp_x, temp_y)) = jacrev(self.neural_net, argnums=(1, 2, 3))(params, t, x, y)

        u_hessian = hessian(self.u_net, argnums=(2, 3))(params, t, x, y)
        v_hessian = hessian(self.v_net, argnums=(2, 3))(params, t, x, y)
        temp_hessian = hessian(self.temp_net, argnums=(2, 3))(params, t, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        temp_xx = temp_hessian[0][0]
        temp_yy = temp_hessian[1][1]

        # phi = 2 * u_x**2 + 2 * v_y**2 + (u_y + v_x)**2
        # phi = u_x ** 2 + u_y ** 2 + v_x ** 2 + v_y ** 2

        ru = u_t + u * u_x + v * u_y + p_x - self.alpha1 * (u_xx + u_yy)
        rv = v_t + u * v_x + v * v_y + p_y - self.alpha1 * (v_xx + v_yy) - self.alpha2 * temp
        rc = u_x + v_y
        re = temp_t + u * temp_x + v * temp_y - self.alpha4 * (temp_xx + temp_yy)

        return ru, rv, rc, re

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        ru_pred, rv_pred, rc_pred, re_pred = self.r_pred_fn(params, t_sorted, batch[:, 1], batch[:, 2])

        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        rv_pred = rv_pred.reshape(self.num_chunks, -1)
        rc_pred = rc_pred.reshape(self.num_chunks, -1)
        re_pred = re_pred.reshape(self.num_chunks, -1)

        ru_l = jnp.mean(ru_pred**2, axis=1)
        rv_l = jnp.mean(rv_pred**2, axis=1)
        rc_l = jnp.mean(rc_pred**2, axis=1)
        re_l = jnp.mean(re_pred**2, axis=1)

        ru_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ ru_l)))
        rv_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rv_l)))
        rc_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rc_l)))
        re_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ re_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([ru_gamma, rv_gamma, rc_gamma, re_gamma])
        gamma = gamma.min(0)

        return ru_l, rv_l, rc_l, re_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Unpack batch
        ics_batch = batch["ics"]
        bcs_batch = batch["bcs"]
        res_batch = batch["res"]

        # Initial condition loss
        coords_batch, u_batch, v_batch, p_batch, temp_batch = ics_batch

        # Initial conditions loss
        u_ic_pred = self.u_ic_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])
        v_ic_pred = self.v_ic_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])
        p_ic_pred = self.p_ic_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])
        temp_ic_pred = self.temp_ic_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])

        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)
        p_ic_loss = jnp.mean((p_ic_pred - p_batch) ** 2)
        temp_ic_loss = jnp.mean((temp_ic_pred - temp_batch) ** 2)

        # Boundary condition losses
        u_bc_pred = self.u_bc_pred_fn(params, bcs_batch[:, 0], bcs_batch[:, 1], bcs_batch[:, 2])
        v_bc_pred = self.v_bc_pred_fn(params, bcs_batch[:, 0], bcs_batch[:, 1], bcs_batch[:, 2])
        temp_bc_pred = self.temp_bc_pred_fn(params, bcs_batch[:, 0], bcs_batch[:, 1], bcs_batch[:, 2])

        u_bc_loss = jnp.mean(u_bc_pred ** 2)
        v_bc_loss = jnp.mean(v_bc_pred ** 2)
        temp_bc_loss = jnp.mean(temp_bc_pred ** 2)

        # t_star = jnp.linspace(self.t_star[0], self.t_star[-1], 64)
        # x_star = jnp.linspace(0, 1, 256)
        #
        # u_bc1_pred = vmap(vmap(self.u_net, (None, None, 0, None)), (None, 0, None, None))(params, self.t_star, x_star, 0.0)
        # u_bc2_pred = vmap(vmap(self.u_net, (None, None, 0, None)), (None, 0, None, None))(params, self.t_star, x_star, 2.0)
        #
        # v_bc1_pred = vmap(vmap(self.v_net, (None, None, 0, None)), (None, 0, None, None))(params, self.t_star, x_star, 0.0)
        # v_bc2_pred = vmap(vmap(self.v_net, (None, None, 0, None)), (None, 0, None, None))(params, self.t_star, x_star, 2.0)
        #
        # temp_bc1_pred = vmap(vmap(self.temp_net, (None, None, 0, None)), (None, 0, None, None))(params, self.t_star, x_star, 0.0)
        # temp_bc2_pred = vmap(vmap(self.temp_net, (None, None, 0, None)), (None, 0, None, None))(params, self.t_star, x_star, 2.0)
        #
        # u_bc_loss = jnp.mean(u_bc1_pred ** 2) + jnp.mean(u_bc2_pred ** 2)
        # v_bc_loss = jnp.mean(v_bc1_pred ** 2) + jnp.mean(v_bc2_pred ** 2)
        # temp_bc_loss = jnp.mean(temp_bc1_pred ** 2) + jnp.mean(temp_bc2_pred ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            ru_l, rv_l, rc_l, re_l, gamma = self.res_and_w(params, res_batch)
            ru_loss = jnp.mean(ru_l * gamma)
            rv_loss = jnp.mean(rv_l * gamma)
            rc_loss = jnp.mean(rc_l * gamma)
            re_loss = jnp.mean(re_l * gamma)

        else:
            ru_pred, rv_pred, rc_pred, re_pred = self.r_pred_fn(
                params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            # Compute loss
            ru_loss = jnp.mean(ru_pred**2)
            rv_loss = jnp.mean(rv_pred**2)
            rc_loss = jnp.mean(rc_pred**2)
            re_loss = jnp.mean(re_pred**2)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "temp_ic": temp_ic_loss,
            "u_bc": u_bc_loss,
            "v_bc": v_bc_loss,
            "temp_bc": temp_bc_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
            "re": re_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, coords, u_ref, v_ref, temp_ref):
        u_pred = self.u_pred_fn(params, t, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, t, coords[:, 0], coords[:, 1])
        temp_pred = self.temp_pred_fn(params, t, coords[:, 0], coords[:, 1])

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        temp_error = jnp.linalg.norm(temp_pred - temp_ref) / jnp.linalg.norm(temp_ref)

        return u_error, v_error, temp_error


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, t, coords, u_ref, v_ref, temp_ref):
        u_error, v_error, temp_error = self.model.compute_l2_error(
            params,
            t, coords,
            u_ref,
            v_ref,
            temp_ref,
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
        self.log_dict["temp_error"] = temp_error

    def __call__(self, state, batch, t, coords, u_ref, v_ref, temp_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, t, coords, u_ref, v_ref, temp_ref)

        if self.config.weighting.use_causal:
            _, _, _, _, causal_weight = self.model.res_and_w(state.params, batch['res'])
            self.log_dict["cas_weight"] = causal_weight.min()

        return self.log_dict
