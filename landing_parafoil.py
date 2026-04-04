# dynamaxsys/landing_parafoil.py
import jax.numpy as jnp
from dynamaxsys.base import Dynamics
from typing import Dict, Optional


def getInertialToBodyRotationMatrix(phi, theta, psi):
    return jnp.array(
        [
            [
                jnp.cos(theta) * jnp.cos(psi),
                jnp.cos(theta) * jnp.sin(psi),
                -jnp.sin(theta),
            ],
            [
                jnp.sin(phi) * jnp.sin(theta) * jnp.cos(psi)
                - jnp.cos(phi) * jnp.sin(psi),
                jnp.sin(phi) * jnp.sin(theta) * jnp.sin(psi)
                + jnp.cos(phi) * jnp.cos(psi),
                jnp.sin(phi) * jnp.cos(theta),
            ],
            [
                jnp.cos(phi) * jnp.sin(theta) * jnp.cos(psi)
                + jnp.sin(phi) * jnp.sin(psi),
                jnp.cos(phi) * jnp.sin(theta) * jnp.sin(psi)
                - jnp.sin(phi) * jnp.cos(psi),
                jnp.cos(phi) * jnp.cos(theta),
            ],
        ]
    )


def getBodyRatesFromEulerRates(phi, theta):
    return jnp.array(
        [
            [1.0, 0.0, -jnp.sin(theta)],
            [0.0, jnp.cos(phi), jnp.sin(phi) * jnp.cos(theta)],
            [0.0, -jnp.sin(phi), jnp.cos(phi) * jnp.cos(theta)],
        ]
    )


def wind_vector_field(z, wind_params: Dict):
    altitude = jnp.maximum(-z, 1.0)

    u_ref = wind_params["u_ref"]
    dir0 = wind_params["dir0"]
    shear_exp = wind_params["shear_exp"]

    z_ref = 10.0
    V_mean = u_ref * (altitude / z_ref) ** shear_exp

    wx = V_mean * jnp.cos(dir0)
    wy = V_mean * jnp.sin(dir0)
    wz = 0.02 * V_mean * jnp.sin(altitude / 200.0)

    return jnp.array([wx, wy, wz])


SPIRAL_RATE = 0.25


class JannParafoil3DOF(Dynamics):
    state_dim: int = 12
    control_dim: int = 1
    disturbance_dim: int = 3

    def __init__(
        self,
        params: dict,
        wind_params: dict,
        failure_mode: Optional[str] = None,
    ):
        object.__setattr__(self, "params", dict(params))
        object.__setattr__(self, "wind_params", dict(wind_params))
        object.__setattr__(self, "failure_mode", failure_mode)

        def dynamics_func(state, control, disturbance=[0, 0, 0], time=0):
            x, y, z, u, v, w, phi, theta, psi, p, q, r = state
            control = jnp.atleast_1d(control)
            delta_a = control[0]

            p_ = self.params
            u_0 = p_["u_0"]
            w_0 = p_["w_0"]
            K_psi = p_["K_psi"]
            T_psi = p_["T_psi"]
            failure_mode = p_.get("failure_mode", self.failure_mode)

            if failure_mode == "straight":
                r_dot = -r / T_psi
                psi_dot = 0.0
            elif failure_mode == "spiral":
                r_dot = 0.0
                psi_dot = SPIRAL_RATE
            else:
                r_dot = (K_psi * delta_a - r) / T_psi
                psi_dot = r

            u = u_0
            v = 0.0
            w = w_0
            phi = 0.0
            theta = 0.0

            R_ib = getInertialToBodyRotationMatrix(phi, theta, psi)
            R_bi = R_ib.T

            wind = wind_vector_field(z, self.wind_params)
            xyz_dot = R_bi @ jnp.array([u, v, w]) + wind

            uvw_dot = jnp.array([0.0, 0.0, 0.0])
            euler_dot = jnp.array([0.0, 0.0, psi_dot])
            pqr_dot = jnp.array([0.0, 0.0, r_dot])

            return jnp.concatenate([xyz_dot, uvw_dot, euler_dot, pqr_dot])

        super().__init__(dynamics_func, self.state_dim, self.control_dim)


class JannParafoil4DOF(Dynamics):
    state_dim: int = 4
    control_dim: int = 2
    disturbance_dim: int = 3

    def __init__(self, params: dict, wind_params: dict, failure_mode: Optional[str] = None):
        object.__setattr__(self, "params", dict(params))
        object.__setattr__(self, "wind_params", dict(wind_params))
        object.__setattr__(self, "failure_mode", failure_mode)

        def dynamics_func(state, control, disturbance=[0, 0, 0], time=0):
            x_pos, y_pos, z_pos, psi = state
            control = jnp.atleast_1d(control)
            delta_a = control[0] if control.size >= 1 else 0.0
            delta_s = control[1] if control.size >= 2 else 0.0

            p_ = self.params
            u_0 = p_.get("u_0", 10.0)
            w_0 = p_.get("w_0", 2.0)
            K_psi = p_.get("K_psi", 0.5)
            T_psi = p_.get("T_psi", 1.0)
            failure_mode = p_.get("failure_mode", self.failure_mode)

            if failure_mode == "straight":
                psi_dot = 0.0
            elif failure_mode == "spiral":
                psi_dot = SPIRAL_RATE
            else:
                psi_dot = K_psi * delta_a / max(T_psi, 1e-6)

            wind = wind_vector_field(z_pos, self.wind_params)
            dx = jnp.array(
                [
                    u_0 * jnp.cos(psi) + wind[0],
                    u_0 * jnp.sin(psi) + wind[1],
                    w_0 + wind[2],
                    psi_dot + 0.0 * delta_s,
                ]
            )
            return dx

        super().__init__(dynamics_func, self.state_dim, self.control_dim)


class JannParafoil4DOF2(Dynamics):
    state_dim: int = 12
    control_dim: int = 2
    disturbance_dim: int = 3

    def __init__(self, params: dict, wind_params: dict, failure_mode: Optional[str] = None):
        object.__setattr__(self, "params", dict(params))
        object.__setattr__(self, "wind_params", dict(wind_params))
        object.__setattr__(self, "failure_mode", failure_mode)

        def dynamics_func(state, control, disturbance=[0, 0, 0], time=0):
            x, y, z, u, v, w, phi, theta, psi, p, q, r = state
            control = jnp.atleast_1d(control)
            delta_a = control[0] if control.size >= 1 else 0.0
            delta_s = control[1] if control.size >= 2 else 0.0

            p_ = self.params
            m = p_.get("m", 1.0)
            S = p_.get("S", 1.0)
            C_L0 = p_.get("C_L0", 0.5)
            C_D0 = p_.get("C_D0", 0.2)
            C_L_delta_s = p_.get("C_L_delta_s", 0.0)
            C_D_delta_s = p_.get("C_D_delta_s", 0.0)
            K_phi = p_.get("K_phi", 0.5)
            T_phi = p_.get("T_phi", 1.0)
            g = p_.get("g", 9.81)

            rho = 1.225
            V_a = jnp.sqrt(u**2 + w**2)
            alpha = jnp.arctan2(w, u)
            L = 0.5 * rho * V_a**2 * S * (C_L0 + C_L_delta_s * delta_s)
            D = 0.5 * rho * V_a**2 * S * (C_D0 + C_D_delta_s * delta_s)

            if self.failure_mode == "straight":
                phi_dot = -phi / T_phi
                psi_dot = 0.0
            elif self.failure_mode == "spiral":
                phi_dot = (-phi + 0.3) / T_phi
                psi_dot = SPIRAL_RATE
            else:
                phi_dot = (K_phi * delta_a - phi) / T_phi
                psi_dot = g / max(u, 1e-6) * jnp.tan(phi) + w * phi_dot / (
                    max(u, 1e-6) * jnp.cos(phi)
                )

            theta_dot = 0.0
            v_dot = 0.0

            u_dot = (L * jnp.sin(alpha) - D * jnp.cos(alpha)) / m - w * psi_dot * jnp.sin(phi) / m
            w_dot = (
                (-L * jnp.cos(alpha) - D * jnp.sin(alpha)) / m
                + g * jnp.cos(phi)
                + u * psi_dot * jnp.sin(phi)
            )

            R_ib = getInertialToBodyRotationMatrix(phi, theta, psi)
            R_bi = R_ib.T
            wind = wind_vector_field(z, self.wind_params)

            vel = R_bi @ jnp.array([u, v, w]) + wind

            # enforce downward motion
            vel = vel.at[2].set(jnp.maximum(vel[2], 0.5))

            xyz_dot = vel
            uvw_dot = jnp.array([u_dot, v_dot, w_dot])
            euler_dot = jnp.array([phi_dot, theta_dot, psi_dot])
            pqr_dot = getBodyRatesFromEulerRates(phi, theta) @ euler_dot

            return jnp.concatenate([xyz_dot, uvw_dot, euler_dot, pqr_dot])

        super().__init__(dynamics_func, self.state_dim, self.control_dim)


class SlegersParafoil6DOF(Dynamics):
    state_dim: int = 12
    control_dim: int = 1
    disturbance_dim: int = 3

    def __init__(self, params: dict, wind_params: dict, failure_mode: Optional[str] = None):
        object.__setattr__(self, "params", dict(params))
        object.__setattr__(self, "wind_params", dict(wind_params))
        object.__setattr__(self, "failure_mode", failure_mode)

        def dynamics_func(state, control, disturbance=[0, 0, 0], time=0):
            x, y, z, u, v, w, phi, theta, psi, p, q, r = state
            control = jnp.atleast_1d(control)
            delta_a = control[0]

            p_ = self.params
            m = p_.get("m", 1.0)
            mmoi = p_.get("mmoi", jnp.eye(3))
            mmoi_inverse = jnp.linalg.inv(mmoi)
            S = p_.get("S", 1.0)
            b = p_.get("b", 1.0)
            c = p_.get("c", 1.0)
            g = p_.get("g", 9.81)
            rho = p_.get("rho", 1.225)

            C_L0 = p_.get("C_L0", 0.5)
            C_L_alpha = p_.get("C_L_alpha", 0.0)
            C_L_delta_a = p_.get("C_L_delta_a", 0.0)
            C_D0 = p_.get("C_D0", 0.2)
            C_D_alpha2 = p_.get("C_D_alpha2", 0.0)
            C_D_delta_a = p_.get("C_D_delta_a", 0.0)

            C_lphi = p_.get("C_lphi", 0.0)
            C_lp = p_.get("C_lp", 0.0)
            C_l_delta_a = p_.get("C_l_delta_a", 0.0)
            C_m0 = p_.get("C_m0", 0.0)
            C_m_alpha = p_.get("C_m_alpha", 0.0)
            C_mq = p_.get("C_mq", 0.0)
            C_n_r = p_.get("C_n_r", 0.0)
            C_n_delta_a = p_.get("C_n_delta_a", 0.0)

            skew_symmetric_pqr = jnp.array([[0.0, -r, q], [r, 0.0, -p], [-q, p, 0.0]])

            alpha = jnp.arctan2(w, u)
            V_a = jnp.sqrt(u**2 + v**2 + w**2)

            C_L = C_L0 + C_L_alpha * alpha + C_L_delta_a * delta_a
            C_D = C_D0 + C_D_alpha2 * alpha**2 + C_D_delta_a * delta_a

            L_div_Va = 0.5 * rho * V_a * S * C_L
            D_div_Va = 0.5 * rho * V_a * S * C_D

            aero_force = L_div_Va * jnp.array([w, 0.0, -u]) - D_div_Va * jnp.array([u, v, w])

            aero_moment = (
                0.5
                * rho
                * V_a**2
                * S
                * jnp.array(
                    [
                        C_lphi * phi * b + C_lp * b**2 * p / (2.0 * max(V_a, 1e-6)) + C_l_delta_a * delta_a * b,
                        C_m0 * c + C_m_alpha * c * alpha + C_mq * c**2 * q / (2.0 * max(V_a, 1e-6)),
                        C_n_r * b**2 * r / (2.0 * max(V_a, 1e-6)) + C_n_delta_a * delta_a * b,
                    ]
                )
            )

            if self.failure_mode == "straight":
                aero_moment = jnp.array([0.0, 0.0, 0.0])
            elif self.failure_mode == "spiral":
                aero_moment = aero_moment + jnp.array([0.0, 0.0, 0.05])

            weight_force = m * g * jnp.array(
                [
                    -jnp.sin(theta),
                    jnp.cos(theta) * jnp.sin(phi),
                    jnp.cos(theta) * jnp.cos(phi),
                ]
            )

            wind = wind_vector_field(z, self.wind_params)

            R_ib = getInertialToBodyRotationMatrix(phi, theta, psi)
            R_bi = R_ib.T
            xyz_dot = R_bi @ jnp.array([u, v, w]) + wind

            uvw_dot_force = (aero_force + weight_force) / m
            uvw_dot_coriolis = -skew_symmetric_pqr @ jnp.array([u, v, w])
            uvw_dot = uvw_dot_force + uvw_dot_coriolis

            pqr_to_euler_dot = jnp.array(
                [
                    [1.0, jnp.sin(phi) * jnp.tan(theta), jnp.cos(phi) * jnp.tan(theta)],
                    [0.0, jnp.cos(phi), -jnp.sin(phi)],
                    [0.0, jnp.sin(phi) / jnp.cos(theta), jnp.cos(phi) / jnp.cos(theta)],
                ]
            )
            euler_dot = pqr_to_euler_dot @ jnp.array([p, q, r])

            pqr_dot = mmoi_inverse @ (
                aero_moment - skew_symmetric_pqr @ mmoi @ jnp.array([p, q, r])
            )

            return jnp.concatenate([xyz_dot, uvw_dot, euler_dot, pqr_dot])

        super().__init__(dynamics_func, self.state_dim, self.control_dim)
