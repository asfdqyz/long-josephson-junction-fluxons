from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time


# ============================================================
# RCSJ + inductive coupling + external flux per cell, open chain
# ============================================================
def system(t, y, N, alpha, i, l, a):
    """
    Open chain, nodes k=0..N-1, cells m=0..N-2 between k and k+1.

    Normalized equations:
        dphi_k/dt = psi_k
        dpsi_k/dt = i - sin(phi_k) - alpha*psi_k - coupling_k

    a[m] = 2*pi*Phi_e[m]/Phi0 is external flux in cell m.
    In the runs below, a is usually zero: fluxons are introduced through
    the initial phase profile, not through external flux.
    """
    phi = y[:N]
    psi = y[N:]

    # a can be None / scalar / array / function a(t, N-1)
    if callable(a):
        a = a(t, N - 1)
    if a is None:
        a = np.zeros(N - 1, dtype=float)
    else:
        a = np.asarray(a, dtype=float)
        if a.ndim == 0:
            a = np.full(N - 1, float(a))
    if a.shape != (N - 1,):
        raise ValueError(f"a must have shape ({N-1},), got {a.shape}")

    dphi_dt = psi
    dpsi_dt = np.zeros_like(psi)

    # left boundary k = 0
    dpsi_dt[0] = (
        i
        - np.sin(phi[0])
        - alpha * psi[0]
        - (phi[0] - phi[1] + a[0]) / l
    )

    # bulk k = 1..N-2
    if N > 2:
        dpsi_dt[1:-1] = (
            i
            - np.sin(phi[1:-1])
            - alpha * psi[1:-1]
            + (phi[2:] + phi[:-2] - 2.0 * phi[1:-1] + a[:-1] - a[1:]) / l
        )

    # right boundary k = N-1
    dpsi_dt[-1] = (
        i
        - np.sin(phi[-1])
        - alpha * psi[-1]
        - (phi[-1] - phi[-2] - a[-1]) / l
    )

    return np.concatenate([dphi_dt, dpsi_dt])


# ============================================================
# Fluxon initial conditions
# ============================================================
def _stable_kink_phi(xi):
    """Stable version of 4*atan(exp(xi))."""
    xi = np.asarray(xi, dtype=float)
    phi = np.empty_like(xi, dtype=float)

    mask = xi < 0.0
    phi[mask] = 4.0 * np.arctan(np.exp(xi[mask]))
    phi[~mask] = 2.0 * np.pi - 4.0 * np.arctan(np.exp(-xi[~mask]))

    return phi


def _stable_sech(xi):
    """Stable sech(xi) = 1/cosh(xi)."""
    xi = np.asarray(xi, dtype=float)
    z = np.exp(-np.abs(xi))
    return 2.0 * z / (1.0 + z * z)


def make_fluxon_ic(N, l, x0, v=0.9, direction="right"):
    """
    Continuum sine-Gordon 2*pi kink placed on the discrete lattice.

    N         -- number of nodes
    l         -- coupling parameter; lattice step dx = sqrt(l)
    x0        -- fluxon center in continuous coordinate x
    v         -- initial fluxon velocity, 0 <= v < 1
    direction -- "left" or "right"

    For a right-moving fluxon:
        phi(x,t) = 4 atan(exp((x - x0 - v*t)/sqrt(1-v^2)))
        psi(x,0) = dphi/dt < 0 near the center
    """
    if not (0.0 <= v < 1.0):
        raise ValueError("Need 0 <= v < 1")

    dx = np.sqrt(l)
    x = dx * np.arange(N)

    gamma = np.sqrt(1.0 - v**2)
    xi = (x - x0) / gamma

    phi0 = _stable_kink_phi(xi)

    if direction == "left":
        sgn = +1.0
    elif direction == "right":
        sgn = -1.0
    else:
        raise ValueError("direction must be 'left' or 'right'")

    psi0 = sgn * (2.0 * v / gamma) * _stable_sech(xi)

    return phi0, psi0


def make_multi_fluxon_y0(
    N,
    l,
    n_fluxons,
    fluxon_cell=1,
    v0=0.9,
    direction="right",
    placement_mode="same",       # "same", "spread", or "custom"
    spacing_cells=8,
    fluxon_cells=None,
):
    """
    Make initial conditions with a controllable number of fluxons.

    n_fluxons:
        Number of 2*pi kinks to put into the initial phase profile.

    placement_mode="same":
        All fluxons are placed at the same center. This imitates the
        Ustinov/Malomed/Sakai style for a 4*pi, 6*pi, ... initial kink:
        two 2*pi kinks at the same x0 produce an initial 4*pi kink.

    placement_mode="spread":
        Fluxons are placed in different cells:
            fluxon_cell, fluxon_cell + spacing_cells, ...

    placement_mode="custom":
        Use the explicit list fluxon_cells, for example [5, 20, 35].

    fluxon_cell and fluxon_cells are 1-based cell-like indices.
    """
    if n_fluxons < 0 or int(n_fluxons) != n_fluxons:
        raise ValueError("n_fluxons must be a non-negative integer")
    n_fluxons = int(n_fluxons)

    dx = np.sqrt(l)

    if n_fluxons == 0:
        phi0 = np.zeros(N)
        psi0 = np.zeros(N)
        return np.concatenate([phi0, psi0]), [], 0.0

    if placement_mode == "same":
        cells = [int(fluxon_cell)] * n_fluxons
    elif placement_mode == "spread":
        cells = [int(fluxon_cell + j * spacing_cells) for j in range(n_fluxons)]
    elif placement_mode == "custom":
        if fluxon_cells is None:
            raise ValueError("For placement_mode='custom', pass fluxon_cells")
        if len(fluxon_cells) != n_fluxons:
            raise ValueError("len(fluxon_cells) must be equal to n_fluxons")
        cells = [int(c) for c in fluxon_cells]
    else:
        raise ValueError("placement_mode must be 'same', 'spread', or 'custom'")

    if min(cells) < 0 or max(cells) > N - 1:
        raise ValueError(
            f"Fluxon centers are outside the chain: cells={cells}. "
            f"Use cells in the approximate range 0..{N-1}."
        )

    phi0 = np.zeros(N, dtype=float)
    psi0 = np.zeros(N, dtype=float)

    for cell in cells:
        x0 = float(cell) * dx
        phi_k, psi_k = make_fluxon_ic(
            N=N,
            l=l,
            x0=x0,
            v=v0,
            direction=direction,
        )
        phi0 += phi_k
        psi0 += psi_k

    y0 = np.concatenate([phi0, psi0])
    q_fluxon = (phi0[-1] - phi0[0]) / (2.0 * np.pi)

    return y0, cells, q_fluxon


# ============================================================
# Voltage extraction
# ============================================================
def avg_voltage_global(sol, N, transient_fraction=0.6):
    """
    Normalized voltage:
        V = <dphi/dt>, averaged over nodes and over the final time window.
    """
    cut = int(transient_fraction * sol.t.size)
    psi = sol.y[N:, cut:]
    return float(np.mean(psi))


def avg_voltage_single_node(sol, N, transient_fraction=0.6, node_index=None):
    if node_index is None:
        node_index = N // 2
    cut = int(transient_fraction * sol.t.size)
    psi = sol.y[N:, cut:]
    return float(np.mean(psi[node_index, :]))


# ============================================================
# I-V scan for arbitrary initial condition
# ============================================================
def simulate_IV(
    N,
    alpha,
    l,
    y0,
    currents,
    a=None,
    t_end=1500.0,
    n_eval=5000,
    method="RK45",
    transient_fraction=0.8,
    voltage_mode="node",       # "global" or "node"
    node_index=None,
    reset_each_current=False,
    show_progress=True,
    label="",
):
    """
    Return V(I) for a current sweep.

    reset_each_current=False:
        Continuation scan: the final state at current I_k is used as the
        initial state for I_{k+1}. This is the usual way to trace an IV branch.

    reset_each_current=True:
        Each current point starts again from exactly the same y0.
    """
    t_span = (0.0, t_end)
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)

    y_init = y0.copy()
    V = []

    t0 = time.time()
    total = len(currents)

    def progress_line(k, i_val):
        if not show_progress:
            return
        done = k + 1
        frac = done / total
        bar_len = 28
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - t0
        rate = elapsed / done
        eta = rate * (total - done)
        print(
            f"\r{label}[{bar}] {done:>4}/{total:<4}  i={i_val: .4f}  "
            f"elapsed={elapsed:7.1f}s  eta={eta:7.1f}s",
            end="",
            flush=True,
        )

    for k, i_val in enumerate(currents):
        if reset_each_current:
            y_init = y0.copy()

        sol = solve_ivp(
            system,
            t_span,
            y_init,
            args=(N, alpha, float(i_val), l, a),
            t_eval=t_eval,
            method=method,
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed at i={i_val}: {sol.message}")

        if voltage_mode == "global":
            V.append(avg_voltage_global(sol, N, transient_fraction))
        elif voltage_mode == "node":
            V.append(avg_voltage_single_node(sol, N, transient_fraction, node_index))
        else:
            raise ValueError("voltage_mode must be 'global' or 'node'")

        y_init = sol.y[:, -1].copy()
        progress_line(k, float(i_val))

    if show_progress:
        print()

    return np.asarray(V)


# ============================================================
# Worker: one IV curve for one number of fluxons
# ============================================================
def run_one_fluxon_number(job):
    (
        n_fluxons,
        N,
        alpha,
        l,
        currents,
        a,
        fluxon_cell,
        v0,
        direction,
        placement_mode,
        spacing_cells,
        fluxon_cells_map,
        t_end,
        n_eval,
        method,
        transient_fraction,
        voltage_mode,
        node_index,
        reset_each_current,
    ) = job

    custom_cells = None
    if fluxon_cells_map is not None and int(n_fluxons) in fluxon_cells_map:
        custom_cells = fluxon_cells_map[int(n_fluxons)]

    actual_placement_mode = placement_mode
    if custom_cells is not None:
        actual_placement_mode = "custom"

    y0, cells, q_fluxon = make_multi_fluxon_y0(
        N=N,
        l=l,
        n_fluxons=int(n_fluxons),
        fluxon_cell=fluxon_cell,
        v0=v0,
        direction=direction,
        placement_mode=actual_placement_mode,
        spacing_cells=spacing_cells,
        fluxon_cells=custom_cells,
    )

    V = simulate_IV(
        N=N,
        alpha=alpha,
        l=l,
        y0=y0,
        currents=currents,
        a=a,
        t_end=t_end,
        n_eval=n_eval,
        method=method,
        transient_fraction=transient_fraction,
        voltage_mode=voltage_mode,
        node_index=node_index,
        reset_each_current=reset_each_current,
        show_progress=False,
    )

    return {
        "n_fluxons": int(n_fluxons),
        "cells": cells,
        "q_fluxon_initial": float(q_fluxon),
        "V": V,
    }


# ============================================================
# Helpers
# ============================================================
def make_current_array(i_min, i_max, di, sweep_direction="up"):
    if di <= 0:
        raise ValueError("di must be positive")
    if sweep_direction == "up":
        return np.arange(i_min, i_max + 0.5 * di, di)
    if sweep_direction == "down":
        return np.arange(i_max, i_min - 0.5 * di, -di)
    raise ValueError("sweep_direction must be 'up' or 'down'")


def safe_float_tag(x):
    return f"{float(x):g}".replace(".", "p").replace("-", "m")



if __name__ == "__main__":

    N = 100
    alpha = 0.1
    l = 0.001

  
    f_ext = 0.0
    a = 2.0 * np.pi * f_ext

    # -------------------------
    # Fluxon initial conditions
    # -------------------------
    fluxon_counts = [1, 2, 3, 4, 5, 6]

    fluxon_cell = 1
    v0 = 0.9
    direction = "right"

    placement_mode = "same"
    spacing_cells = 12

    
    fluxon_cells_map = None

   
    i_min = 0.0
    i_max = 0.8
    di = 0.001

   
    sweep_direction = "up"   
    currents = make_current_array(i_min, i_max, di, sweep_direction=sweep_direction)

    t_end = 1500.0
    n_eval = 5000
    method = "RK45"             
    transient_fraction = 0.8

    voltage_mode = "node"
    node_index = N // 2

    reset_each_current = True


    cpu = os.cpu_count() or 2
    max_workers = min(len(fluxon_counts), max(1, cpu - 1))

    jobs = [
        (
            int(n_fluxons),
            N,
            alpha,
            l,
            currents,
            a,
            fluxon_cell,
            v0,
            direction,
            placement_mode,
            spacing_cells,
            fluxon_cells_map,
            t_end,
            n_eval,
            method,
            transient_fraction,
            voltage_mode,
            node_index,
            reset_each_current,
        )
        for n_fluxons in fluxon_counts
    ]

    print(
        f"Running {len(jobs)} IV curves in parallel with max_workers={max_workers}: "
        f"N_fluxons={fluxon_counts}"
    )
    print(
        f"N={N}, alpha={alpha}, l={l}, sweep={sweep_direction}, "
        f"I range: {currents[0]:g} -> {currents[-1]:g}, points={len(currents)}"
    )
    print(f"placement_mode={placement_mode}, fluxon_cell={fluxon_cell}, v0={v0}, direction={direction}")

    t0 = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one_fluxon_number, job) for job in jobs]
        for done_count, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            n_fluxons = res["n_fluxons"]
            results[n_fluxons] = res
            elapsed = time.time() - t0
            print(
                f"Done {done_count}/{len(jobs)}: "
                f"N_fl={n_fluxons}, cells={res['cells']}, "
                f"initial winding={res['q_fluxon_initial']:.6g}, elapsed={elapsed:.1f}s"
            )


    out_dir = Path("iv_fluxon_results")
    out_dir.mkdir(exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = (
        f"IV_multi_fluxons_parallel_"
        f"N{N}_alpha{safe_float_tag(alpha)}_l{safe_float_tag(l)}_"
        f"I{safe_float_tag(i_min)}to{safe_float_tag(i_max)}_dI{safe_float_tag(di)}_"
        f"{sweep_direction}_{placement_mode}_{stamp}"
    )
    data_path = out_dir / f"{base}.csv"
    png_path = out_dir / f"{base}.png"

    n_sorted = sorted(results)
    data_columns = [currents]
    header_cols = ["I_normalized"]
    for n_fluxons in n_sorted:
        data_columns.append(results[n_fluxons]["V"])
        header_cols.append(f"V_normalized_Nfl_{n_fluxons}")

    data = np.column_stack(data_columns)
    np.savetxt(
        data_path,
        data,
        delimiter=",",
        header=",".join(header_cols),
        comments="",
    )

    plt.figure(figsize=(8, 6))

    for n_fluxons in n_sorted: 
        V = results[n_fluxons]["V"] 
        cells = results[n_fluxons]["cells"] 
        label = rf"$N_{{fl}}={n_fluxons}$, cells={cells}" 
        plt.plot(V, currents, "o", ms=3, lw=1.0, label=label)

    plt.xlabel(r"$V = \langle \dot{\phi} \rangle$ (normalized)")
    plt.ylabel(r"$I/I_c$ (normalized)")
    plt.title("I-V characteristics for different initial fluxon numbers")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")

    print("Saved data:", data_path.resolve())
    print("Saved plot:", png_path.resolve())

    plt.show()
