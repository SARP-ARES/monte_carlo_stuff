import os
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Callable, List, Tuple

# -------------------------
# Repo bootstrap
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

from dynamaxsys.landing_parafoil import JannParafoil3DOF

# -------------------------
# Run a single simulation
# -------------------------
def run_sim(model_class: Callable, params: Dict, wind_params: Dict, u_profile: float,
            T: float = 50.0, dt: float = 0.1) -> Tuple[float, float]:
    model = model_class(params=params, wind_params=wind_params)
    x0 = jnp.zeros(model.state_dim)
    u_seq = jnp.ones(int(T / dt)) * u_profile

    def scan_fn(x, u):
        dx = model(x, u)
        x_next = x + dx * dt
        return x_next, x_next

    final_state, states = jax.lax.scan(scan_fn, x0, u_seq)
    x, y = final_state[0], final_state[1]
    return float(x), float(y)


# -------------------------
# Monte Carlo sweep
# -------------------------
def monte_carlo(param_name, param_values, model_class, base_params, wind_params):
    results = {}

    for val in param_values:
        val_float = float(val)  # ← FIX

        wind_params_sweep = wind_params.copy()
        wind_params_sweep[param_name] = val_float

        x, y = run_sim(model_class, base_params, wind_params_sweep, u_profile=val_float)

        results[val_float] = (x, y)  # ← FIX
        print(f"{param_name}={val_float:.3f} -> x={x:.3f}, y={y:.3f}")

    return results


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    base_params = {"u_0": 2.0, "w_0": 0.0, "K_psi": 0.5, "T_psi": 1.0}
    wind_params = {"u_ref": 2.0, "v_ref": 0.0, "h_scale": 100.0}
    u_vals = list(map(float, jnp.linspace(1.0, 3.0, 5)))

    results = monte_carlo(
        param_name="u_ref",
        param_values=u_vals,
        model_class=JannParafoil3DOF,
        base_params=base_params,
        wind_params=wind_params
    )

    print("Monte Carlo results:", results)

#-------------------------
# PLotting results
#-------------------------
def plot_results(results: Dict[float, Tuple[float, float]], param_name: str):
    """
    Plot landing position vs swept parameter
    """
    params = list(results.keys())
    xs = [results[p][0] for p in params]
    ys = [results[p][1] for p in params]

    plt.figure()
    
    # X landing vs parameter
    plt.subplot(2, 1, 1)
    plt.plot(params, xs, marker='o')
    plt.xlabel(param_name)
    plt.ylabel("Landing X")
    plt.grid(True)

    # Y landing vs parameter
    plt.subplot(2, 1, 2)
    plt.plot(params, ys, marker='o')
    plt.xlabel(param_name)
    plt.ylabel("Landing Y")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_results(results, "u_ref")
