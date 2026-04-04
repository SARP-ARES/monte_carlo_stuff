# monte_carlo_landing.py
import os
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# -------------------------
# Repo bootstrap (keep it)
# -------------------------
root = Path(os.getcwd()).resolve()
repo_root = None
for parent in [root] + list(root.parents):
    if (parent / "dynamaxsys" / "__init__.py").exists():
        repo_root = parent
        break

if repo_root is None:
    raise RuntimeError("Could not locate repo root")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dynamaxsys.landing_parafoil import (
    JannParafoil3DOF,
    JannParafoil4DOF,
    JannParafoil4DOF2,
    SlegersParafoil6DOF,
)

MODEL_MAP = {
    "3dof": JannParafoil3DOF,
    "4dof": JannParafoil4DOF,
    "4dof2": JannParafoil4DOF2,
    "6dof": SlegersParafoil6DOF,
}

WIND_BOUNDS = {
    "u_ref": (2.5, 12.0),
    "dir0": (np.deg2rad(225.0), np.deg2rad(250.0)),
    "shear_exp": (0.08, 0.25),
}

PARAM_DESCRIPTIONS = {
    "u_ref": "Reference wind speed at 10 m altitude (m/s)",
    "dir0": "Wind direction at reference altitude (radians, from inertial x-axis)",
    "shear_exp": "Vertical wind shear exponent (dimensionless, power-law profile)",
}

def default_initial_state(state_dim: int, altitude_m: float = 1000.0):
    if state_dim == 12:
        return jnp.array([0.0, 0.0, -altitude_m] + [0.0] * 9)
    if state_dim == 4:
        return jnp.array([0.0, 0.0, -altitude_m, 0.0])
    if state_dim == 6:
        return jnp.array([0.0, 0.0, -altitude_m, 0.0, 0.0, 0.0])
    raise ValueError(f"Unsupported state_dim={state_dim}")


def run_sim(
    model_class,
    params,
    wind_params,
    control,
    x0=None,
    dt=0.1,
    max_steps=2000,
    failure_mode=None,
):
    model = model_class(params=params, wind_params=wind_params, failure_mode=failure_mode)

    if x0 is None:
        x = default_initial_state(model.state_dim, altitude_m=1000.0)
    else:
        x = jnp.asarray(x0)

    prev_z = x[2]

    for _ in range(max_steps):
        dx = model(x, control)
        x = x + dx * dt

        # landing condition
        if x[2] >= 0.0:
            break

        # safety: if we are going UP or stuck → stop
        if x[2] > prev_z:
            break

        prev_z = x[2]

    return float(x[0]), float(x[1])


def sample_wind_params(rng: np.random.Generator):
    return {
        "u_ref": float(rng.uniform(*WIND_BOUNDS["u_ref"])),
        "dir0": float(rng.uniform(*WIND_BOUNDS["dir0"])),
        "shear_exp": float(rng.uniform(*WIND_BOUNDS["shear_exp"])),
    }


def monte_carlo(
    param_name,
    values,
    n_samples,
    model_class,
    base_params,
    control,
    x0=None,
    dt=0.1,
    max_steps=5000,
    failure_mode=None,
    seed=0,
):
    rng = np.random.default_rng(seed)
    results = {}

    print(f"\n--- Sweeping parameter: {param_name} ---")
    print(f"Meaning: {PARAM_DESCRIPTIONS.get(param_name, 'Unknown parameter')}\n")

    for val in values:
        landings = []

        for _ in range(n_samples):
            wind_params = sample_wind_params(rng)
            wind_params[param_name] = float(val)

            x, y = run_sim(
                model_class=model_class,
                params=base_params,
                wind_params=wind_params,
                control=control,
                x0=x0,
                dt=dt,
                max_steps=max_steps,
                failure_mode=failure_mode,
            )
            landings.append((x, y))

        results[float(val)] = landings
        if param_name == "dir0":
            print(f"{param_name}={np.rad2deg(val):.1f} deg complete")
        else:
            print(f"{param_name}={float(val):.3f} complete")

    return results


def plot_results(results, param_name):
    plt.figure()
    for val, pts in results.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.scatter(xs, ys, alpha=0.25, label=f"{val:.2f}")

    plt.xlabel("X landing (m)")
    plt.ylabel("Y landing (m)")
    desc = PARAM_DESCRIPTIONS.get(param_name, param_name)
    plt.title(f"Landing dispersion vs {param_name}\n({desc})")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    model_class = JannParafoil3DOF

    base_params = {
        "u_0": 10.0,
        "w_0": 2.0,   # positive downward, since z starts negative and lands at z >= 0
        "K_psi": 0.5,
        "T_psi": 1.0,
        "failure_mode": None,
    }

    control = 0.0
    n_samples = 100

    # Choose one: None, "straight", "spiral"
    failure_mode = None

    # Factor study examples
    u_vals = np.linspace(2.5, 12.0, 5)
    dir_vals = np.deg2rad(np.linspace(225.0, 250.0, 5))
    shear_vals = np.linspace(0.08, 0.25, 5)

    results_u = monte_carlo(
        param_name="u_ref",
        values=u_vals,
        n_samples=n_samples,
        model_class=model_class,
        base_params=base_params,
        control=control,
        failure_mode=failure_mode,
        seed=1,
    )
    plot_results(results_u, "u_ref")

    results_dir = monte_carlo(
        param_name="dir0",
        values=dir_vals,
        n_samples=n_samples,
        model_class=model_class,
        base_params=base_params,
        control=control,
        failure_mode=failure_mode,
        seed=2,
    )
    plot_results(results_dir, "dir0")

    results_shear = monte_carlo(
        param_name="shear_exp",
        values=shear_vals,
        n_samples=n_samples,
        model_class=model_class,
        base_params=base_params,
        control=control,
        failure_mode=failure_mode,
        seed=3,
    )
    plot_results(results_shear, "shear_exp")
