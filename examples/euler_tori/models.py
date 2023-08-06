from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional

import flax
from flax import linen as nn
from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, random, tree_map, jacfwd, jacrev
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

import optax

from jaxpi import archs
from jaxpi.models import ForwardIVP
from jaxpi.utils import jacobian_fn, ntk_fn
from jaxpi.evaluator import BaseEvaluator

from utils import u0_v0_rho0


class Euler(ForwardIVP):
    def __init__(self, config, t, xy, u0, v0, rho0):
        super().__init__(config)

        self.t = t
        self.xy = xy
        self.u0 = u0
        self.v0 = v0
        self.rho0 = rho0

        # Predictions over a grid
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        t = t / self.t[-1]  # scale time to [0, 1]

        z = jnp.stack([t, x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        rho = outputs[3]
        return u, v, p, rho

    def u_net(self, params, t, x, y):
        u, _, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _, _ = self.neural_net(params, t, x, y)
        return v

    def p_net(self, params, t, x, y):
        _, _, p, _ = self.neural_net(params, t, x, y)
        return p

    def rho_net(self, params, t, x, y):
        _, _, _, rho = self.neural_net(params, t, x, y)
        return rho

    def r_net(self, params, t, x, y):
        u, v, p, rho = self.neural_net(params, t, x, y)

        u_t = grad(self.u_net, argnums=1)(params, t, x, y)
        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)

        v_t = grad(self.v_net, argnums=1)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y)

        p_x = grad(self.p_net, argnums=2)(params, t, x, y)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y)

        rho_t = grad(self.rho_net, argnums=1)(params, t, x, y)
        rho_x = grad(self.rho_net, argnums=2)(params, t, x, y)
        rho_y = grad(self.rho_net, argnums=3)(params, t, x, y)

        ru = rho * (u_t + u * u_x + v * u_y + p_x)
        rv = rho * (v_t + u * v_x + v * v_y + p_y)
        # cont = rho_t + rho * u_x + u * rho_x + rho * v_y + v * rho_y  # Can be reduced to the following line, since u_x + v_y = 0
        rc = rho_t + u * rho_x + v * rho_y
        rd = u_x + v_y

        return ru, rv, rc, rd

    def ru_net(self, params, t, x, y):
        ru, _, _, _ = self.r_net(params, t, x, y)
        return ru

    def rv_net(self, params, t, x, y):
        _, rv, _, _ = self.r_net(params, t, x, y)
        return rv

    def rc_net(self, params, t, x, y):
        _, _, rc, _ = self.r_net(params, t, x, y)
        return rc

    def rd_net(self, params, t, x, y):
        _, _, _, rd = self.r_net(params, t, x, y)
        return rd

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        t_sorted = batch[:, 0].sort()
        ru_pred, rv_pred, rc_pred, rd_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1], batch[:, 2]
        )

        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        rv_pred = rv_pred.reshape(self.num_chunks, -1)
        rc_pred = rc_pred.reshape(self.num_chunks, -1)
        rd_pred = rd_pred.reshape(self.num_chunks, -1)

        l_ru = jnp.mean(ru_pred**2, axis=1)
        l_rv = jnp.mean(rv_pred**2, axis=1)
        l_rc = jnp.mean(rc_pred**2, axis=1)
        l_rd = jnp.mean(rd_pred**2, axis=1)

        gamma_ru = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_ru)))
        gamma_rv = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_rv)))
        gamma_rc = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_rc)))
        gamma_rd = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_rd)))

        gamma = jnp.vstack([gamma_ru, gamma_rv, gamma_rc, gamma_rd])
        gamma = gamma.min(0)

        return l_ru, l_rv, l_rc, l_rd, gamma

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        u_ic_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(self.u_net, params, 0.0, self.xy[:, 0], self.xy[:, 1])
        v_ic_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(self.v_net, params, 0.0, self.xy[:, 0], self.xy[:, 1])
        rho_ic_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(self.rho_net, params, 0.0, self.xy[:, 0], self.xy[:, 1])

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            # sort the time step for causal loss
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1], batch[:, 2]]).T
            ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.ru_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rv_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rv_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rd_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rd_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )

            ru_ntk = ru_ntk.reshape(self.num_chunks, -1)
            rv_ntk = rv_ntk.reshape(self.num_chunks, -1)
            rc_ntk = rc_ntk.reshape(self.num_chunks, -1)
            rd_ntk = rd_ntk.reshape(self.num_chunks, -1)

            ru_ntk = jnp.mean(ru_ntk, axis=1)
            rv_ntk = jnp.mean(rv_ntk, axis=1)
            rc_ntk = jnp.mean(rc_ntk, axis=1)
            rd_ntk = jnp.mean(rd_ntk, axis=1)

            _, _, _, _, casual_weights = self.res_and_w(params, batch)
            ru_ntk = ru_ntk * casual_weights
            rv_ntk = rv_ntk * casual_weights
            rc_ntk = rc_ntk * casual_weights
            rd_ntk = rd_ntk * casual_weights

        else:
            ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.ru_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rv_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rv_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rd_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rd_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )

        ntk_dict = {
            "u_ic": u_ic_ntk,
            "v_ic": v_ic_ntk,
            "rho_ic": rho_ic_ntk,
            "ru": ru_ntk,
            "rv": rv_ntk,
            "rc": rc_ntk,
            "rd": rd_ntk,
        }

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition losses
        u0_pred = vmap(self.u_net, (None, None, 0, 0))(
            params, 0.0, self.xy[:, 0], self.xy[:, 1]
        )
        v0_pred = vmap(self.v_net, (None, None, 0, 0))(
            params, 0.0, self.xy[:, 0], self.xy[:, 1]
        )
        rho0_pred = vmap(self.rho_net, (None, None, 0, 0))(
            params, 0.0, self.xy[:, 0], self.xy[:, 1]
        )

        # Compute loss
        u_ic_loss = jnp.mean((u0_pred - self.u0) ** 2)
        v_ic_loss = jnp.mean((v0_pred - self.v0) ** 2)
        rho_ic_loss = jnp.mean((rho0_pred - self.rho0) ** 2)

        # Residual losses
        if self.config.weighting.use_causal == True:
            ru_l, rv_l, rc_l, rd_l, gamma = self.res_and_w(params, batch)
            ru_loss = jnp.mean(ru_l * gamma)
            rv_loss = jnp.mean(rv_l * gamma)
            rc_loss = jnp.mean(rc_l * gamma)
            rd_loss = jnp.mean(rd_l * gamma)
        else:
            ru_pred, rv_pred, rc_pred, rd_pred = self.r_pred_fn(
                params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            # Compute loss
            ru_loss = jnp.mean(ru_pred**2)
            rv_loss = jnp.mean(rv_pred**2)
            rc_loss = jnp.mean(rc_pred**2)
            rd_loss = jnp.mean(rd_pred**2)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "rho_ic": rho_ic_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
            "rd": rd_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, coords, u_ref, v_ref, rho_ref):
        u_pred = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))(
            params, t, coords[:, 0], coords[:, 1]
        )
        v_pred = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))(
            params, t, coords[:, 0], coords[:, 1]
        )
        rho_pred = vmap(vmap(self.rho_net, (None, None, 0, 0)), (None, 0, None, None))(
            params, t, coords[:, 0], coords[:, 1]
        )

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        rho_error = jnp.linalg.norm(rho_pred - rho_ref) / jnp.linalg.norm(rho_ref)

        return u_error, v_error, rho_error


class EulerEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, t, coords, u_ref, v_ref, rho_ref):
        u_error, v_error, rho_error = self.model.compute_l2_error(
            params, t, coords, u_ref, v_ref, rho_ref
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
        self.log_dict["rho_error"] = rho_error

    def log_preds(self):
        pass

    def __call__(self, state, batch, t, coords, u_ref, v_ref, rho_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, t, coords, u_ref, v_ref, rho_ref)

        if self.config.weighting.use_causal:
            _, _, _, _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_preds:
            self.log_preds()

        return self.log_dict
