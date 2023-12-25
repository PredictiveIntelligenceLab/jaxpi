from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn


class NavierStokes2D(ForwardBVP):
    def __init__(
        self,
        config,
        u_inflow,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        Re,
    ):
        super().__init__(config)

        self.u_in = u_inflow  # inflow profile
        self.Re = Re  # Reynolds number

        # Initialize coordinates
        self.inflow_coords = inflow_coords
        self.outflow_coords = outflow_coords
        self.wall_coords = wall_coords
        self.cylinder_coords = cylinder_coords
        self.noslip_coords = jnp.vstack((self.wall_coords, self.cylinder_coords))

        # Non-dimensionalized domain length and width
        self.L, self.W = self.noslip_coords.max(axis=0) - self.noslip_coords.min(axis=0)

        # Predict functions over batch
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))

    def neural_net(self, params, x, y):
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]
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

    def r_net(self, params, x, y):
        u, v, p = self.neural_net(params, x, y)

        (u_x, u_y), (v_x, v_y), (p_x, p_y) = jacrev(self.neural_net, argnums=(1, 2))(params, x, y)

        u_hessian = hessian(self.u_net, argnums=(1, 2))(params, x, y)
        v_hessian = hessian(self.v_net, argnums=(1, 2))(params, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        # PDE residual
        ru = u * u_x + v * u_y + p_x - (u_xx + u_yy) / self.Re
        rv = u * v_x + v * v_y + p_y - (v_xx + v_yy) / self.Re
        rc = u_x + v_y

        # outflow boundary residual
        u_out = u_x - p
        v_out = v_x

        return ru, rv, rc, u_out, v_out

    def ru_net(self, params, x, y):
        ru, _, _, _, _ = self.r_net(params, x, y)
        return ru

    def rv_net(self, params, x, y):
        _, rv, _, _, _ = self.r_net(params, x, y)
        return rv

    def rc_net(self, params, x, y):
        _, _, rc, _, _ = self.r_net(params, x, y)
        return rc

    def u_out_net(self, params, x, y):
        _, _, _, u_out, _ = self.r_net(params, x, y)
        return u_out

    def v_out_net(self, params, x, y):
        _, _, _, _, v_out = self.r_net(params, x, y)
        return v_out

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Inflow boundary conditions
        u_in_pred = self.u_pred_fn(
            params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )
        v_in_pred = self.v_pred_fn(
            params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )

        u_in_loss = jnp.mean((u_in_pred - self.u_in) ** 2)
        v_in_loss = jnp.mean(v_in_pred**2)

        # Outflow boundary conditions
        _, _, _, u_out_pred, v_out_pred = self.r_pred_fn(
            params, self.outflow_coords[:, 0], self.outflow_coords[:, 1]
        )

        u_out_loss = jnp.mean(u_out_pred**2)
        v_out_loss = jnp.mean(v_out_pred**2)

        # No-slip boundary conditions
        u_noslip_pred = self.u_pred_fn(
            params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )
        v_noslip_pred = self.v_pred_fn(
            params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )

        u_noslip_loss = jnp.mean(u_noslip_pred**2)
        v_noslip_loss = jnp.mean(v_noslip_pred**2)

        # Residual losses
        ru_pred, rv_pred, rc_pred, _, _ = self.r_pred_fn(
            params, batch[:, 0], batch[:, 1]
        )

        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            "u_in": u_in_loss,
            "v_in": v_in_loss,
            "u_out": u_out_loss,
            "v_out": v_out_loss,
            "u_noslip": u_noslip_loss,
            "v_noslip": v_noslip_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        u_in_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_net, params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )
        v_in_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.v_net, params, self.inflow_coords[:, 0], self.inflow_coords[:, 1]
        )

        u_out_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_out_net, params, self.outflow_coords[:, 0], self.outflow_coords[:, 1]
        )
        v_out_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.v_out_net, params, self.outflow_coords[:, 0], self.outflow_coords[:, 1]
        )

        u_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_net, params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )
        v_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.v_net, params, self.noslip_coords[:, 0], self.noslip_coords[:, 1]
        )

        ru_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.ru_net, params, batch[:, 0], batch[:, 1]
        )
        rv_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.rv_net, params, batch[:, 0], batch[:, 1]
        )
        rc_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.rc_net, params, batch[:, 0], batch[:, 1]
        )

        ntk_dict = {
            "u_in": u_in_ntk,
            "v_in": v_in_ntk,
            "u_out": u_out_ntk,
            "v_out": v_out_ntk,
            "u_noslip": u_noslip_ntk,
            "v_noslip": v_noslip_ntk,
            "ru": ru_ntk,
            "rv": rv_ntk,
            "rc": rc_ntk,
        }

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, coords, u_test, v_test):
        u_pred = self.u_pred_fn(params, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, coords[:, 0], coords[:, 1])

        u_error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        v_error = jnp.linalg.norm(v_pred - v_test) / jnp.linalg.norm(v_test)

        return u_error, v_error

    def u_v_grads(self, params, x, y):
        u_x = grad(self.u_net, argnums=1)(params, x, y)
        v_x = grad(self.v_net, argnums=1)(params, x, y)

        u_y = grad(self.u_net, argnums=2)(params, x, y)
        v_y = grad(self.v_net, argnums=2)(params, x, y)

        return u_x, v_x, u_y, v_y

    def compute_drag_lift(self, params, U_star, L_star):
        nu = 0.001  # Dimensional viscosity
        radius = 0.05  # radius of cylinder
        center = (0.2, 0.2)  # center of cylinder
        num_theta = 256  # number of points on cylinder for evaluation

        # Discretize cylinder into points
        theta = jnp.linspace(0.0, 2 * jnp.pi, num_theta)
        d_theta = theta[1] - theta[0]
        ds = radius * d_theta

        # Cylinder coordinates
        x_cyl = radius * jnp.cos(theta) + center[0]
        y_cyl = radius * jnp.sin(theta) + center[1]

        # Out normals of cylinder
        n_x = jnp.cos(theta)
        n_y = jnp.sin(theta)

        # Nondimensionalize input cylinder coordinates
        x_cyl = x_cyl / L_star
        y_cyl = y_cyl / L_star

        # Nondimensionalize fonrt and back points
        front = jnp.array([center[0] - radius, center[1]]) / L_star
        back = jnp.array([center[0] + radius, center[1]]) / L_star

        # Predictions
        u_x_pred, v_x_pred, u_y_pred, v_y_pred = vmap(self.u_v_grads, (None, 0, 0))(
            params, x_cyl, y_cyl
        )
        p_pred = vmap(self.p_net, (None, 0, 0))(params, x_cyl, y_cyl)

        p_front_pred = self.p_net(params, front[0], front[1])
        p_back_pred = self.p_net(params, back[0], back[1])
        p_diff = p_front_pred - p_back_pred

        # Dimensionalize velocity gradients and pressure
        u_x_pred = u_x_pred * U_star / L_star
        v_x_pred = v_x_pred * U_star / L_star
        u_y_pred = u_y_pred * U_star / L_star
        v_y_pred = v_y_pred * U_star / L_star
        p_pred = p_pred * U_star**2
        p_diff = p_diff * U_star**2

        I0 = (-p_pred[:-1] + 2 * nu * u_x_pred[:-1]) * n_x[:-1] + nu * (
            u_y_pred[:-1] + v_x_pred[:-1]
        ) * n_y[:-1]
        I1 = (-p_pred[1:] + 2 * nu * u_x_pred[1:]) * n_x[1:] + nu * (
            u_y_pred[1:] + v_x_pred[1:]
        ) * n_y[1:]

        F_D = 0.5 * jnp.sum(I0 + I1) * ds

        I0 = (-p_pred[:-1] + 2 * nu * v_y_pred[:-1]) * n_y[:-1] + nu * (
            u_y_pred[:-1] + v_x_pred[:-1]
        ) * n_x[:-1]
        I1 = (-p_pred[1:] + 2 * nu * v_y_pred[1:]) * n_y[1:] + nu * (
            u_y_pred[1:] + v_x_pred[1:]
        ) * n_x[1:]

        F_L = 0.5 * jnp.sum(I0 + I1) * ds

        # Nondimensionalized drag and lift and pressure difference
        C_D = 2 / (U_star**2 * L_star) * F_D
        C_L = 2 / (U_star**2 * L_star) * F_L

        return C_D, C_L, p_diff


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, coords, u_ref, v_ref):
        u_error, v_error = self.model.compute_l2_error(params, coords, u_ref, v_ref)
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error

    # def log_preds(self, params, x_star, y_star):
    #     u_pred = vmap(vmap(model.u_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     v_pred = vmap(vmap(model.v_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     U_pred = jnp.sqrt(u_pred ** 2 + v_pred ** 2)
    #
    #     fig = plt.figure()
    #     plt.pcolor(U_pred.T, cmap='jet')
    #     log_dict['U_pred'] = fig
    #     fig.close()

    def __call__(self, state, batch, coords, u_ref, v_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, coords, u_ref, v_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params, coords)

        return self.log_dict
