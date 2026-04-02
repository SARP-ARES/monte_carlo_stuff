# =========================
# IMPORT BOOTSTRAP (KEEP THIS)
# =========================
import os
import sys
from pathlib import Path

root = Path(os.getcwd()).resolve()
repo_root = None
for parent in [root] + list(root.parents):
    if (parent / "dynamaxsys" / "__init__.py").exists():
        repo_root = parent
        break

if repo_root is None:
    raise RuntimeError(
        "Could not locate the repository root containing `dynamaxsys`."
    )

repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

for m in ["dynamaxsys", "dynamaxsys.parafoil"]:
    if m in sys.modules:
        del sys.modules[m]

# =========================
# NORMAL IMPORTS
# =========================
import jax.numpy as jnp
import time

from dynamaxsys.base import get_discrete_time_dynamics
from dynamaxsys.parafoil import JannParafoil3DOF
from model_parameters import jann_3dof_params
import controllers

from simulator import (
    simulate_with_controller,
    DEG_TO_RAD,
    plot_3D_traj,
    plot_states,
)

# =========================
# SIMULATION SETTINGS
# =========================
dt = 0.01
time_horizon = 100
N = int(time_horizon / dt)
ts = jnp.arange(0, time_horizon, dt)

# =========================
# INITIAL STATE
# =========================
jann_3dof_initial_state = {
    "x_ned": -300.0,
    "y_ned": 200.0,
    "z_ned": -500.0,
    "u": jann_3dof_params["u_0"],
    "v": 0.0,
    "w": jann_3dof_params["w_0"],
    "phi": 0.0,
    "theta": 0.0,
    "psi": 120.0 * DEG_TO_RAD,
    "p": 0.0,
    "q": 0.0,
    "r": 0.0,
}

x0 = jnp.array(list(jann_3dof_initial_state.values()))

# disturbances (unused since wind is internal)
ds = jnp.zeros((N, 3))

# =========================
# SWEEP PARAMETERS
# =========================
wind_ids = [1, 2, 3, 4, 5]
failure_modes = [None, "straight", "spiral"]

results = []

# =========================
# MONTE CARLO LOOP
# =========================
for wind_id in wind_ids:
    for failure_mode in failure_modes:

        print(f"\nRunning: wind={wind_id}, failure={failure_mode}")

        # Build dynamics with parameters
        dyn = JannParafoil3DOF(
            jann_3dof_params,
            wind_scenario_id=wind_id,
            failure_mode=failure_mode,
        )

        dyn_discrete = get_discrete_time_dynamics(dyn, dt)

        # RESET controller state each run
        ctrl_state0 = (
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(False),
        )

        controller = controllers.TwelveStateHeadingController(
            kp=1,
            ki=0.0,
            kd=0.1,
            max_output=1.0,
            min_output=-1.0,
            rate_limit=0.2,
            spiral_mode=True,
            inner_spiral_range=10.0,
            outer_spiral_range=100.0,
        )

        start = time.time()

        xs, us = simulate_with_controller(
            x0,
            dyn_discrete,
            controller,
            ctrl_state0,
            ds,
            ts,
        )

        end = time.time()
        print(f"Runtime: {end - start:.3f}s")

        results.append(
            {
                "wind_id": wind_id,
                "failure_mode": failure_mode,
                "xs": xs,
                "us": us,
            }
        )

# =========================
# PLOTTING
# =========================
import matplotlib.pyplot as plt

# =========================
# ORGANIZE RESULTS
# =========================
from collections import defaultdict

grouped = defaultdict(list)
for r in results:
    grouped[r["wind_id"]].append(r)

# =========================
# STATES TO PLOT
# =========================
label_idx = {
    "z": 2,
    "u": 3,
    "w": 5,
    "psi": 8,
    "r": 11,
}

state_labels = list(label_idx.keys())
state_indices = list(label_idx.values())

failure_modes = [None, "straight", "spiral"]
failure_titles = ["Nominal", "Straight Failure", "Spiral Failure"]

# =========================
# PLOTTING
# =========================
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================
# GROUP RESULTS
# =========================
grouped = defaultdict(list)
for r in results:
    grouped[r["wind_id"]].append(r)

# =========================
# STATES
# =========================
label_idx = {
    "z": 2,
    "u": 3,
    "w": 5,
    "psi": 8,
    "r": 11,
}

failure_modes = [None, "straight", "spiral"]
failure_titles = ["Nominal", "Straight Failure", "Spiral Failure"]

# =========================
# PLOTTING
# =========================
for wind_id, wind_results in grouped.items():

    n_rows = len(label_idx) + 1  # +1 for 3D traj row
    n_cols = len(failure_modes)

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Wind Scenario {wind_id}", fontsize=16)

    axes = []

    # -------------------------
    # CREATE AXES GRID
    # -------------------------
    for row in range(n_rows):
        row_axes = []
        for col in range(n_cols):

            if row == 0:
                # 3D plot row
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection='3d')
            else:
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)

            row_axes.append(ax)
        axes.append(row_axes)

    # -------------------------
    # FILL PLOTS
    # -------------------------
    for col, failure_mode in enumerate(failure_modes):

        result = next(r for r in wind_results if r["failure_mode"] == failure_mode)
        xs = result["xs"]

        # ---------- 3D TRAJECTORY ----------
        ax3d = axes[0][col]

        x = xs[:, 0]
        y = xs[:, 1]
        z = xs[:, 2]

        ax3d.plot(x, y, z)
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")

        ax3d.set_box_aspect([1,1,1])
        ax3d.scatter(x[0], y[0], z[0], marker='o')  # start
        ax3d.scatter(x[-1], y[-1], z[-1], marker='x')  # end

        if col == 0:
            ax3d.set_title(failure_titles[col])

        # ---------- STATES ----------
        for row, (label, idx) in enumerate(label_idx.items(), start=1):

            ax = axes[row][col]

            # fix dimension mismatch
            ydata = xs[:-1, idx]

            # convert angles to degrees
            if label in ["psi", "r"]:
                ydata = ydata * 180 / jnp.pi

            ax.plot(ts, ydata)

            if col == 0:
                ax.set_ylabel(label)

            if row == n_rows - 1:
                ax.set_xlabel("Time (s)")

            ax.grid(True)

    plt.tight_layout()
    plt.show()
