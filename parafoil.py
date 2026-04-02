import jax.numpy as jnp
from dynamaxsys.base import Dynamics


def getInertialToBodyRotationMatrix(phi, theta, psi):
    """
    Computes the rotation matrix to transform linear velocity vectors from the inertial frame to body frame
    given the roll (phi), pitch (theta), and yaw (psi) angles.

    Args:
        phi (float): Roll angle in radians.
        theta (float): Pitch angle in radians.
        psi (float): Yaw angle in radians.

    Returns:
        jnp.ndarray: A 3x3 rotation matrix.
    """
    inertial_to_body = jnp.array(
        [
            [  # row 1
                jnp.cos(theta) * jnp.cos(psi),
                jnp.cos(theta) * jnp.sin(psi),
                -jnp.sin(theta),
            ],
            [  # row 2
                jnp.sin(phi) * jnp.sin(theta) * jnp.cos(psi)
                - jnp.cos(phi) * jnp.sin(psi),
                jnp.sin(phi) * jnp.sin(theta) * jnp.sin(psi)
                + jnp.cos(phi) * jnp.cos(psi),
                jnp.sin(phi) * jnp.cos(theta),
            ],
            [  # row 3
                jnp.cos(phi) * jnp.sin(theta) * jnp.cos(psi)
                + jnp.sin(phi) * jnp.sin(psi),
                jnp.cos(phi) * jnp.sin(theta) * jnp.sin(psi)
                - jnp.sin(phi) * jnp.cos(psi),
                jnp.cos(phi) * jnp.cos(theta),
            ],
        ]
    )

    return inertial_to_body

def getBodyRatesFromEulerRates(phi, theta):
    """
    Computes the transformation matrix to convert Euler angle rates (phi_dot, theta_dot, psi_dot)
    to body frame angular rates (p, q, r).

    Args:
        phi (float): Roll angle in radians.
        theta (float): Pitch angle in radians.

    Returns:
        jnp.ndarray: A 3x3 transformation matrix.
    """
    pqr_from_euler_rates = jnp.array(
        [
            [1, 0, -jnp.sin(theta)],
            [0, jnp.cos(phi),jnp.sin(phi)*jnp.cos(theta)],
            [0, -jnp.sin(phi), jnp.cos(phi)*jnp.cos(theta)],
        ]
    )
    return pqr_from_euler_rates

#WIND_SCENARIO_ID = 4  # 1 = best/calmest, 5 = worst/most severe

WIND_SCENARIOS = {
    1: {
        "name": "best",
        "u_ref": 2.5,
        "shear_exp": 0.08,
        "dir0": jnp.deg2rad(250.0),
        "veer_max": jnp.deg2rad(8.0),
        "veer_scale": 1400.0,
        "layer_alt": 1500.0,
        "layer_amp": jnp.deg2rad(2.0),
        "layer_width": 450.0,
        "turb_ref": 0.20,
        "turb_decay": 900.0,
        "vertical_ratio": 0.01,
        "phase_xy": 0.0,
        "phase_z": 0.0,
    },
    2: {
        "name": "mild",
        "u_ref": 4.0,
        "shear_exp": 0.10,
        "dir0": jnp.deg2rad(245.0),
        "veer_max": jnp.deg2rad(15.0),
        "veer_scale": 1200.0,
        "layer_alt": 1200.0,
        "layer_amp": jnp.deg2rad(4.0),
        "layer_width": 400.0,
        "turb_ref": 0.35,
        "turb_decay": 700.0,
        "vertical_ratio": 0.02,
        "phase_xy": 0.7,
        "phase_z": 0.3,
    },
    3: {
        "name": "nominal",
        "u_ref": 6.0,
        "shear_exp": 0.15,
        "dir0": jnp.deg2rad(240.0),
        "veer_max": jnp.deg2rad(25.0),
        "veer_scale": 1000.0,
        "layer_alt": 900.0,
        "layer_amp": jnp.deg2rad(7.0),
        "layer_width": 300.0,
        "turb_ref": 0.55,
        "turb_decay": 500.0,
        "vertical_ratio": 0.03,
        "phase_xy": 1.4,
        "phase_z": 0.9,
    },
    4: {
        "name": "adverse",
        "u_ref": 9.0,
        "shear_exp": 0.20,
        "dir0": jnp.deg2rad(235.0),
        "veer_max": jnp.deg2rad(40.0),
        "veer_scale": 850.0,
        "layer_alt": 800.0,
        "layer_amp": jnp.deg2rad(12.0),
        "layer_width": 250.0,
        "turb_ref": 0.85,
        "turb_decay": 350.0,
        "vertical_ratio": 0.05,
        "phase_xy": 2.0,
        "phase_z": 1.3,
    },
    5: {
        "name": "worst",
        "u_ref": 12.0,
        "shear_exp": 0.25,
        "dir0": jnp.deg2rad(225.0),
        "veer_max": jnp.deg2rad(65.0),
        "veer_scale": 700.0,
        "layer_alt": 650.0,
        "layer_amp": jnp.deg2rad(18.0),
        "layer_width": 180.0,
        "turb_ref": 1.20,
        "turb_decay": 250.0,
        "vertical_ratio": 0.08,
        "phase_xy": 2.7,
        "phase_z": 1.8,
    },
}

# -----------------------------
# FAILURE MODES
# -----------------------------
#FAILURE_MODE = None  
# Options:
# None       → normal flight
# "straight" → no turning authority (ballistic drift)
# "spiral"   → constant turning (autorotation)

SPIRAL_RATE = 0.25  # rad/s (~14 deg/s), tune as needed


def wind_vector_field(z, scenario_id=None):
    """
    Altitude-dependent wind vector field with selectable Mojave scenarios.

    Scenario 1 = best/calmest
    Scenario 5 = worst/most severe
    """

    sid = WIND_SCENARIO_ID if scenario_id is None else scenario_id
    p = WIND_SCENARIOS[int(sid)]

    altitude = jnp.maximum(-z, 1.0)

    # Mean wind speed profile
    z_ref = 10.0
    V_mean = p["u_ref"] * (altitude / z_ref) ** p["shear_exp"]

    # Mean wind direction profile
    theta_mean = (
        p["dir0"]
        + p["veer_max"] * (1.0 - jnp.exp(-altitude / p["veer_scale"]))
        + p["layer_amp"] * jnp.tanh((altitude - p["layer_alt"]) / p["layer_width"])
    )

    # Mean components
    wx_mean = V_mean * jnp.cos(theta_mean)
    wy_mean = V_mean * jnp.sin(theta_mean)
    wz_mean = p["vertical_ratio"] * V_mean * jnp.sin(altitude / 600.0)

    # Smooth turbulence that decays with altitude
    turb_amp = p["turb_ref"] * jnp.exp(-altitude / p["turb_decay"]) + 0.03 * V_mean

    turbulence = turb_amp * jnp.array(
        [
            jnp.sin(altitude / 40.0 + p["phase_xy"]),
            jnp.cos(altitude / 57.0 + 0.5 * p["phase_xy"]),
            0.35 * jnp.sin(altitude / 25.0 + p["phase_z"]),
        ]
    )

    return jnp.array([wx_mean, wy_mean, wz_mean]) + turbulence

class JannParafoil4DOF(Dynamics):
    state_dim: int = 4  # u, w, phi, psi
    control_dim: int = 2  # delta_a, delta_s
    disturbance_dim: int = 3  # wind velocity in x, y, z

    m: float  # mass
    S: float  # parafoil area
    C_L0: float  # baseline lift coefficient
    C_D0: float  # baseline drag coefficient
    C_L_delta_s: float  # lift coefficient per unit deflection of steering line
    C_D_delta_s: float  # drag coefficient per unit deflection of steering line
    K_phi: float  # roll model gain
    T_phi: float  # roll model time constant
    g: float  # gravity constant

    wind_scenario_id: int
    failure_mode: str | None

    def __init__(self, params: dict):
        self.m = params["m"]
        self.S = params["S"]
        self.C_L0 = params["C_L0"]
        self.C_D0 = params["C_D0"]
        self.C_L_delta_s = params["C_L_delta_s"]
        self.C_D_delta_s = params["C_D_delta_s"]
        self.K_phi = params["K_phi"]
        self.T_phi = params["T_phi"]
        self.g = params["g"]

        def dynamics_func(state, control, disturbance=[0,0,0], time=0):
            u, w, phi, psi = state
            delta_a, delta_s = control
            # wind_x_velocity, wind_y_velocity, wind_z_velocity = disturbance #TODO implement disturbances

            # Aerodynamics
            C_L = self.C_L0 + self.C_L_delta_s * delta_s  # Lift coefficient
            C_D = self.C_D0 + self.C_D_delta_s * delta_s  # Drag coefficient
            V_a = jnp.sqrt(u**2 + w**2)  # Airspeed
            alpha = jnp.arctan2(w, u)  # Angle of Attack
            rho = 1.225  # Air density at sea level

            L = 0.5 * rho * V_a**2 * self.S * C_L  # Lift
            D = 0.5 * rho * V_a**2 * self.S * C_D  # Drag

            # Equations of motion
            # -----------------------------
            # FAILURE MODE LOGIC
            # -----------------------------
            if self.failure_mode == "straight":
                phi_dot = -phi / self.T_phi
                psi_dot = 0.0

            elif self.failure_mode == "spiral":
                phi_dot = (-phi + 0.3) / self.T_phi
                psi_dot = SPIRAL_RATE

            else:
                phi_dot = (self.K_phi * delta_a - phi) / self.T_phi
                psi_dot = self.g / u * jnp.tan(phi) + w * phi_dot / (
                    u * jnp.cos(phi)
                )

            u_dot = (
                L * jnp.sin(alpha) - D * jnp.cos(alpha)
            ) / self.m - w * psi_dot * jnp.sin(phi) / self.m  # fwd accel

            w_dot = (
                (-L * jnp.cos(alpha) - D * jnp.sin(alpha)) / self.m
                + self.g * jnp.cos(phi)
                + u * psi_dot * jnp.sin(phi)
            )  # down accel

            return jnp.array([u_dot, w_dot, phi_dot, psi_dot])

        # initialize super class Dynamics object
        super().__init__(dynamics_func, self.state_dim, self.control_dim)


class JannParafoil4DOF2(Dynamics):
    """
    4DOF model by Thomas Jann 
    DOI: https://doi.org/10.2514/6.2001-2016

    Assumes roll angle changes due to asymmetric deflection with a first order delay
    and yaw rate is coupled nonlinearly with roll angle
    """
    state_dim: int = 12  # x, y, z, u, v, w, phi, theta, psi, p, q, r
    control_dim: int = 2  # delta_a, delta_s
    disturbance_dim: int = 3  # wind velocity in x, y, z

    m: float  # mass
    S: float  # parafoil area
    C_L0: float  # baseline lift coefficient
    C_D0: float  # baseline drag coefficient
    C_L_delta_s: float  # lift coefficient per unit deflection of steering line
    C_D_delta_s: float  # drag coefficient per unit deflection of steering line
    K_phi: float  # roll model gain
    T_phi: float  # roll model time constant
    g: float  # gravity constant

    wind_scenario_id: int
    failure_mode: str | None

    def __init__(self, params: dict):
        self.m = params["m"]
        self.S = params["S"]
        self.C_L0 = params["C_L0"]
        self.C_D0 = params["C_D0"]
        self.C_L_delta_s = params["C_L_delta_s"]
        self.C_D_delta_s = params["C_D_delta_s"]
        self.K_phi = params["K_phi"]
        self.T_phi = params["T_phi"]
        self.g = params["g"]

        def dynamics_func(state, control, disturbance=[0,0,0], time=0):
            x, y, z, u, v, w, phi, theta, psi, p, q, r = state
            delta_a, delta_s = control
            wind_x_velocity, wind_y_velocity, wind_z_velocity = disturbance #TODO implement disturbances

            # Aerodynamics
            C_L = self.C_L0 + self.C_L_delta_s * delta_s  # Lift coefficient
            C_D = self.C_D0 + self.C_D_delta_s * delta_s  # Drag coefficient
            V_a = jnp.sqrt(u**2 + w**2)  # Airspeed
            alpha = jnp.arctan2(w, u)  # Angle of Attack
            rho = 1.225  # Air density at sea level

            L = 0.5 * rho * V_a**2 * self.S * C_L  # Lift
            D = 0.5 * rho * V_a**2 * self.S * C_D  # Drag

            # Equations of motion
            # -----------------------------
            # FAILURE MODE LOGIC
            # -----------------------------
            if self.failure_mode == "straight":
                phi_dot = -phi / self.T_phi
                psi_dot = 0.0

            elif self.failure_mode == "spiral":
                phi_dot = (-phi + 0.3) / self.T_phi
                psi_dot = SPIRAL_RATE

            else:
                phi_dot = (self.K_phi * delta_a - phi) / self.T_phi
                psi_dot = self.g / u * jnp.tan(phi) + w * phi_dot / (
                    u * jnp.cos(phi)
                )

            theta_dot = 0.0  # assume no pitch

            u_dot = (
                L * jnp.sin(alpha) - D * jnp.cos(alpha)
            ) / self.m - w * psi_dot * jnp.sin(phi) / self.m  # fwd accel

            v_dot = 0.0  # assume no side velocity

            w_dot = (
                (-L * jnp.cos(alpha) - D * jnp.sin(alpha)) / self.m
                + self.g * jnp.cos(phi)
                + u * psi_dot * jnp.sin(phi)
            )  # down accel

            
            inertial_to_body = getInertialToBodyRotationMatrix(phi, theta, psi)
            body_to_inertial = inertial_to_body.T  # rotation matrix is orthogonal

            # inertial-frame velocity
            wind = wind_vector_field(z, self.wind_scenario_id)

            xyz_dot = body_to_inertial @ jnp.array([u, v, w]) + wind

            uvw_dot = jnp.array([u_dot, v_dot, w_dot])

            euler_dot = jnp.array([phi_dot, theta_dot, psi_dot])

            #get body frame angular velocities from euler rates
            pqr_from_euler_dot = getBodyRatesFromEulerRates(phi, theta)
            pqr_dot = pqr_from_euler_dot @ euler_dot

            return jnp.concatenate([xyz_dot, uvw_dot, euler_dot, pqr_dot])

        # initialize super class Dynamics object
        super().__init__(dynamics_func, self.state_dim, self.control_dim)



class JannParafoil3DOF(Dynamics):
    """
    3DOF model by Thomas Jann 
    DOI: https://doi.org/10.2514/6.2001-2016

    Assumes constant forward velocity and sink rate, yaw rate changes due to asymmetrid deflection with first order delay,
    roll angle, pitch angle, sideslip, and side velocity are all assumed zero.
    
    The only states used in this dynamics model are psi (yaw angle) and r (yaw rate).
    The other states are included for simulation and compatibility with 6DOF models.
    """
    state_dim: int = 12  # x, y, z, u, v, w, phi, theta, psi, p, q, r
    control_dim: int = 1  # delta_a
    disturbance_dim: int = 3  # wind velocity in x, y, z

    u_0: float  # constant forward velocity
    w_0: float  # constant sink rate
    K_psi: float  # psi model gain
    T_psi: float  # psi model time constant

    wind_scenario_id: int
    failure_mode: str | None

    def __init__(self, params: dict, wind_scenario_id=None, failure_mode=None):
        self.u_0 = params["u_0"]
        self.w_0 = params["w_0"]
        self.K_psi = params["K_psi"]
        self.T_psi = params["T_psi"]
        self.wind_scenario_id = wind_scenario_id
        self.failure_mode = failure_mode

        def dynamics_func(state, control, disturbance=[0,0,0], time=0):
            x, y, z, u, v, w, phi, theta, psi, p, q, r = state
            delta_a = control
            wind_x_velocity, wind_y_velocity, wind_z_velocity = disturbance #TODO implement disturbances in Jann models

            # Yaw rate EOM (r = psi_dot)
            # -----------------------------
            # FAILURE MODE LOGIC
            # -----------------------------
            if self.failure_mode == "straight":
                r_dot = -r / self.T_psi
                psi_dot = 0.0

            elif self.failure_mode == "spiral":
                r_dot = 0.0
                psi_dot = SPIRAL_RATE

            else:
                r_dot = (self.K_psi * delta_a - r) / self.T_psi  
                psi_dot = r

            # no roll or pitch angle
            phi = 0.0
            theta = 0.0

            # constant body velocities
            u = self.u_0
            v = 0.0
            w = self.w_0

            inertial_to_body = getInertialToBodyRotationMatrix(phi, theta, psi)
            body_to_inertial = inertial_to_body.T  # rotation matrix is orthogonal
            
            # pqr_from_euler_dot = getBodyRatesFromEulerRates(phi, theta)

            # inertial-frame velocity
            wind = wind_vector_field(z, self.wind_scenario_id)

            xyz_ned_dot = body_to_inertial @ jnp.array([u, v, w]) + wind
            uvw_dot = jnp.array([0, 0, 0])
            euler_dot = jnp.array([0, 0, psi_dot])
            pqr_dot = jnp.array([0, 0, r_dot])  # only r_dot nonzero

            # # body frame rotational dynamics
            # pqr = pqr_from_euler_dot @ euler_dot

            return jnp.concatenate([xyz_ned_dot, uvw_dot, euler_dot, pqr_dot])

        # initialize super class Dynamics object
        super().__init__(dynamics_func, self.state_dim, self.control_dim)




class SlegersParafoil6DOF(Dynamics):
    state_dim: int = 12  # x, y, z, u, v, w, phi, theta, psi, p, q, r
    control_dim: int = 1  # delta_a
    disturbance_dim: int = 3  # wind velocity in x, y, z

    # Physical parameters
    m: float  # mass (kg)
    mmoi: jnp.ndarray  # moment of inertia matrix (3x3)
    mmoi_inverse: jnp.ndarray  # inverse of moment of inertia matrix (3x3)
    S: float  # parafoil area
    b: float  # span
    c: float  # chord length
    g: float  # gravity constant
    rho: float  # air density

    # Lift and drag coefficients
    C_L0: float  # baseline lift coefficient
    C_L_alpha: float  # lift coefficient per radian AoA
    C_L_delta_a: float  # lift coefficient per unit asymmetric deflection
    C_D0: float  # baseline drag coefficient
    C_D_alpha2: float  # quadratic drag coefficient per radian^2 AoA
    C_D_delta_a: float  # drag coefficient per unit asymmetric deflection

    # Aerodynamic moment coefficients
    C_lphi: float  # roll moment coefficient
    C_lp: float  # roll damping coefficient
    C_l_delta_a: float  # roll moment per unit asymmetric deflection
    C_m0: float  # pitch moment coefficient
    C_m_alpha: float  # pitch moment per radian AoA
    C_mq: float  # pitch damping coefficient
    C_n_r: float  # yaw damping coefficient
    C_n_delta_a: float  # yaw moment per unit asymmetric deflection

    wind_scenario_id: int
    failure_mode: str | None

    def __init__(self, params: dict):
        # Physical parameters
        self.m = params["m"]
        self.mmoi = params["mmoi"]
        self.mmoi_inverse = jnp.linalg.inv(self.mmoi)
        self.S = params["S"]
        self.b = params["b"]
        self.c = params["c"]
        self.g = params["g"]
        self.rho = params["rho"]

        # Lift and drag coefficients
        self.C_L0 = params["C_L0"]
        self.C_D0 = params["C_D0"]
        self.C_L_alpha = params["C_L_alpha"]
        self.C_D_alpha2 = params["C_D_alpha2"]
        self.C_L_delta_a = params["C_L_delta_a"]
        self.C_D_delta_a = params["C_D_delta_a"]

        # Aerodynamic moment coefficients
        self.C_lphi = params["C_lphi"]
        self.C_lp = params["C_lp"]
        self.C_l_delta_a = params["C_l_delta_a"]
        self.C_m0 = params["C_m0"]
        self.C_m_alpha = params["C_m_alpha"]
        self.C_mq = params["C_mq"]
        self.C_n_r = params["C_n_r"]
        self.C_n_delta_a = params["C_n_delta_a"]

        def dynamics_func(state, control, disturbance=[0, 0, 0], time=0):
            x, y, z, u, v, w, phi, theta, psi, p, q, r = state
            delta_a = control
            wind_x_velocity, wind_y_velocity, wind_z_velocity = disturbance

            skew_symmetric_pqr = jnp.array([[0, -r, q], [r, 0, -p], [-q, p, 0]])

            # Angle of Attack
            alpha = jnp.arctan2(w, u)  # assumes chordline along fwd-body axis

            # Lift and Drag Coefficients
            C_L = self.C_L0 + self.C_L_alpha * alpha + self.C_L_delta_a * delta_a
            C_D = self.C_D0 + self.C_D_alpha2 * alpha**2 + self.C_D_delta_a * delta_a

            V_a = jnp.sqrt(u**2 + v**2 + w**2)  # Airspeed

            inertial_to_body = getInertialToBodyRotationMatrix(phi, theta, psi)
            body_to_inertial = inertial_to_body.T  # rotation matrix is orthogonal

            L_div_Va = 0.5 * self.rho * V_a * self.S * C_L  # Lift
            D_div_Va = 0.5 * self.rho * V_a * self.S * C_D  # Drag

            # Aerodynamic Force
            aero_force = L_div_Va * jnp.array([w, 0, -u]) - D_div_Va * jnp.array(
                [u, v, w]
            )

            # Aerodynamic Moment
            aero_moment = (
                0.5
                * self.rho
                * V_a**2
                * self.S
                * jnp.array(
                    [
                        self.C_lphi * phi * self.b
                        + self.C_lp * self.b**2 * p / (2 * V_a)
                        + self.C_l_delta_a * delta_a * self.b,
                        self.C_m0 * self.c
                        + self.C_m_alpha * self.c * alpha
                        + self.C_mq * self.c**2 * q / (2 * V_a),
                        self.C_n_r * self.b**2 * r / (2 * V_a)
                        + self.C_n_delta_a * delta_a * self.b,
                    ]
                )
            )

            # -----------------------------
        # FAILURE MODE LOGIC (affects rotational dynamics)
        # -----------------------------
        if self.failure_mode == "straight":
            aero_moment = jnp.array([0.0, 0.0, 0.0])

        elif self.failure_mode == "spiral":
            aero_moment = aero_moment + jnp.array([0.0, 0.0, 0.05])

            # Weight Force
            weight_force = (
                self.m
                * self.g
                * jnp.array(
                    [
                        -jnp.sin(theta),
                        jnp.cos(theta) * jnp.sin(phi),
                        jnp.cos(theta) * jnp.cos(phi),
                    ]
                )
            )

            ## EQUATIONS OF MOTION ##

            # inertial-frame velocity
            wind = wind_vector_field(z, self.wind_scenario_id)

            xyz_dot = body_to_inertial @ jnp.array([u, v, w]) + wind
            # body-frame translation dynamics
            uvw_dot_force = 1 / self.m * (aero_force + weight_force)

            uvw_dot_coriolis = -skew_symmetric_pqr @ jnp.array([u, v, w])

            uvw_dot = uvw_dot_force + uvw_dot_coriolis

            pqr_to_euler_dot = jnp.array(
                [
                    [1, jnp.sin(phi) * jnp.tan(theta), jnp.cos(phi) * jnp.tan(theta)],
                    [0, jnp.cos(phi), -jnp.sin(phi)],
                    [0, jnp.sin(phi) / jnp.cos(theta), jnp.cos(phi) / jnp.cos(theta)],
                ]
            )

            # euler angle dynamics (from body rates)
            euler_dot = pqr_to_euler_dot @ jnp.array([p, q, r])

            # body frame rotational dynamics
            pqr_dot = self.mmoi_inverse @ (
                aero_moment - skew_symmetric_pqr @ self.mmoi @ jnp.array([p, q, r])
            )

            return jnp.concatenate([xyz_dot, uvw_dot, euler_dot, pqr_dot])

        # initialize superclass Dynamics object
        super().__init__(dynamics_func, self.state_dim, self.control_dim)


if __name__ == "__main__":
    # test rotation matrix
    phi = 0 * jnp.pi / 180.0
    theta = 0 * jnp.pi / 180.0
    psi = 90 * jnp.pi / 180.0

    R = getInertialToBodyRotationMatrix(phi, theta, psi)
    print("R (inertial to body):\n", R)
    print("R.T (body to inertial):\n", R.T)
    print("R @ R.T (should be identity):\n", (R @ R.T).round(6))
    
    uvw = jnp.array([1.0, 0.0, 1.0])  # forward in body
    xyz_dot = R.T @ uvw # body to inertial
    print("uvw (body velocity):", uvw)
    print("xyz_dot (inertial velocity):", xyz_dot)
