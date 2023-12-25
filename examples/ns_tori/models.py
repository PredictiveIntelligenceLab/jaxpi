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
    def __init__(self, config, t_star, x_star, y_star, u0, v0, w0, nu):
        super().__init__(config)

        self.u0 = u0
        self.v0 = v0
        self.w0 = w0

        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star

        self.nu = nu

        # Predictions over a grid
        self.u0_pred_fn = vmap(
            vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None)
        )
        self.v0_pred_fn = vmap(
            vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None)
        )
        self.w0_pred_fn = vmap(
            vmap(self.w_net, (None, None, None, 0)), (None, None, 0, None)
        )

        self.u_pred_fn = vmap(
            vmap(vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None)),
            (None, 0, None, None),
        )
        self.v_pred_fn = vmap(
            vmap(vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None)),
            (None, 0, None, None),
        )
        self.w_pred_fn = vmap(
            vmap(vmap(self.w_net, (None, None, None, 0)), (None, None, 0, None)),
            (None, 0, None, None),
        )
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        t = t / self.t_star[-1]
        z = jnp.stack([t, x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        return u, v

    def u_net(self, params, t, x, y):
        u, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v = self.neural_net(params, t, x, y)
        return v

    def w_net(self, params, t, x, y):
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        w = v_x - u_y
        return w

    def r_net(self, params, t, x, y):
        u, v = self.neural_net(params, t, x, y)

        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y)

        w_t, w_x, w_y = jacrev(self.w_net, argnums=(1, 2, 3))(params, t, x, y)

        w_hessian = hessian(self.w_net, argnums=(2, 3))(params, t, x, y)

        w_xx = w_hessian[0][0]
        w_yy = w_hessian[1][1]

        mom = w_t + u * w_x + v * w_y - self.nu * (w_xx + w_yy)
        cont = u_x + v_y

        return mom, cont

    def mom_net(self, params, t, x, y):
        mom, _ = self.r_net(params, t, x, y)
        return mom

    def cont_net(self, params, t, x, y):
        _, cont = self.r_net(params, t, x, y)
        return cont

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        rm_pred, rc_pred = self.r_pred_fn(params, t_sorted, batch[:, 1], batch[:, 2])

        rm_pred = rm_pred.reshape(self.num_chunks, -1)
        rc_pred = rc_pred.reshape(self.num_chunks, -1)

        rm_l = jnp.mean(rm_pred**2, axis=1)
        rc_l = jnp.mean(rc_pred**2, axis=1)

        rm_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rm_l)))
        rc_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rc_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([rm_gamma, rc_gamma])
        gamma = gamma.min(0)

        return rm_l, rc_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial conditions loss
        u0_pred = self.u0_pred_fn(params, 0.0, self.x_star, self.y_star)
        v0_pred = self.v0_pred_fn(params, 0.0, self.x_star, self.y_star)
        w0_pred = self.w0_pred_fn(params, 0.0, self.x_star, self.y_star)

        u0_loss = jnp.mean((u0_pred - self.u0) ** 2)
        v0_loss = jnp.mean((v0_pred - self.v0) ** 2)
        w0_loss = jnp.mean((w0_pred - self.w0) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            rm_l, rc_l, gamma = self.res_and_w(params, batch)
            rm_loss = jnp.mean(rm_l * gamma)
            rc_loss = jnp.mean(rc_l * gamma)

        else:
            rm_pred, rc_pred = self.r_pred_fn(
                params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            # Compute loss
            rm_loss = jnp.mean(rm_pred**2)
            rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            "u_ic": u0_loss,
            "v_ic": v0_loss,
            "w_ic": w0_loss,
            "rm": rm_loss,
            "rc": rc_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        u_ic_ntk = vmap(
            vmap(ntk_fn, (None, None, None, None, 0)), (None, None, None, 0, None)
        )(self.u_net, params, 0.0, self.x_star, self.y_star)
        v_ic_ntk = vmap(
            vmap(ntk_fn, (None, None, None, None, 0)), (None, None, None, 0, None)
        )(self.v_net, params, 0.0, self.x_star, self.y_star)
        w_ic_ntk = vmap(
            vmap(ntk_fn, (None, None, None, None, 0)), (None, None, None, 0, None)
        )(self.w_net, params, 0.0, self.x_star, self.y_star)

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1], batch[:, 2]]).T
            rm_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.mom_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.cont_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )

            rm_ntk = rm_ntk.reshape(self.num_chunks, -1)
            rc_ntk = rc_ntk.reshape(self.num_chunks, -1)

            rm_ntk = jnp.mean(rm_ntk, axis=1)
            rc_ntk = jnp.mean(rc_ntk, axis=1)

            _, _, casual_weights = self.res_and_w(params, batch)
            rm_ntk = rm_ntk * casual_weights
            rc_ntk = rc_ntk * casual_weights
        else:
            rm_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.mom_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.cont_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )

        ntk_dict = {
            "u_ic": u_ic_ntk,
            "v_ic": v_ic_ntk,
            "w_ic": w_ic_ntk,
            "rm": rm_ntk,
            "rc": rc_ntk,
        }
        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, x, y, u_ref, v_ref, w_ref):
        u_pred = self.u_pred_fn(params, t, x, y)
        v_pred = self.v_pred_fn(params, t, x, y)
        w_pred = self.w_pred_fn(params, t, x, y)

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        w_error = jnp.linalg.norm(w_pred - w_ref) / jnp.linalg.norm(w_ref)

        return u_error, v_error, w_error


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, v_ref, w_ref):
        u_error, v_error, w_error = self.model.compute_l2_error(
            params,
            self.model.t_star,
            self.model.x_star,
            self.model.y_star,
            u_ref,
            v_ref,
            w_ref,
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
        self.log_dict["w_error"] = w_error

    def __call__(self, state, batch, u_ref, v_ref, w_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref, v_ref, w_ref)

        if self.config.weighting.use_causal:
            _, _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        return self.log_dict
