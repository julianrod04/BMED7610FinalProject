"""
Hong et al. (2012) Fig 4A for the FT population model, with the real
L5 pyramidal cells (Linaro et al. 2019) as a reference.

Fig 4A plots output correlation rho vs input correlation c, one line per
input mean mu, plus a contour map of rho over (c, mu). Tightly clustered
lines = rho-c relation invariant to the input -> c can be decoded from rho.

The FT filter is mean-subtracted, so mu has no effect on FT (paper Fig 6B).
We therefore make two versions:
  left : literal mu sweep   (sigma^2 = 2000)  -> lines should coincide
  right: sigma^2 sweep      (mu = 0)          -> the knob that changes FT rate
<rho> = mean spike-count correlation over all 190 pairs of 20 FT neurons,
count window T = 40 ms (same as the real-data plots). Real PYR curve:
mean over 15 virtual pairs (6 cells), trial-shuffle corrected.

Run:  python plot_fig4a_ft.py            (uses cache if present)
      python plot_fig4a_ft.py --rerun    (re-simulate)
"""
import sys
import types
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent              # LinaroPaper/
PROJECT = HERE.parent                               # BMED7610FinalProject/
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

# FT_model.py imports numba only for its njit decorator; fall back to plain
# Python if numba is not installed (we use a vectorised detector below anyway).
try:
    import numba  # noqa: F401
except ImportError:
    sys.modules["numba"] = types.SimpleNamespace(njit=lambda f=None, **k: f if f else (lambda g: g))

import FT_model as ft                                       # noqa: E402
from linaro_loader import load_dataset, rho_vs_c            # noqa: E402

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
C_VALUES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
MU_VALUES = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])      # pA, sigma^2 = 2000
MU_SWEEP_VAR = 2000.0
VAR_VALUES = np.array([10.0, 20.0, 50.0, 200.0, 1000.0, 7000.0])  # pA^2, mu = 0; ~10-42 Hz, 20 pA^2 ~ PYR rate
N_NEURONS = 20
SECONDS = 12.0
N_RUNS = 6
WINDOW_MS = 40
SEED = 4401
CACHE = HERE / "fig4a_ft_cache.npz"
OUT_PNG = HERE / "fig4a_FT.png"

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


# ----------------------------------------------------------------------------
# Simulation (reuses FT_model's filter, currents and correlation code)
# ----------------------------------------------------------------------------
def ft_spikes_batch(voltage, threshold=ft.THRESHOLD_MV, ahp_step=ft.AHP_MV,
                    decay=ft.ahp_decay):
    """Vectorised over rows; identical rule to FT_model.ft_spikes."""
    n, steps = voltage.shape
    spikes = np.zeros((n, steps), dtype=np.uint8)
    ahp = np.zeros(n)
    prev = np.full(n, -1e9)
    for i in range(steps):
        ahp *= decay
        v = voltage[:, i]
        fire = (prev < threshold) & (v >= threshold) & (v + ahp >= threshold)
        spikes[fire, i] = 1
        ahp[fire] += ahp_step
        prev = v
    return spikes


def simulate_population(currents):
    """Same as FT_model.simulate_ft_population but all neurons at once."""
    warm = int(round(ft.WARMUP_MS / ft.DT_MS))
    bin_steps = int(round(ft.POP_BIN_MS / ft.DT_MS))
    n_bins = (currents.shape[1] - warm) // bin_steps
    v = lfilter(ft.filter_kernel, [1.0], currents, axis=1)
    raw = ft_spikes_batch(v)[:, warm:warm + n_bins * bin_steps]
    return raw.reshape(currents.shape[0], n_bins, bin_steps).sum(axis=2).astype(np.uint8)


def run_condition(mu, var, c, rng):
    """N_RUNS independent runs (fresh common + private noise each).
    Returns mean <rho>, SEM over runs, mean rate (Hz)."""
    rhos, rates = [], []
    for _ in range(N_RUNS):
        seed = int(rng.integers(0, 2**31 - 1))
        cur = ft.make_population_currents(c, seed, N_NEURONS, SECONDS, var) + mu
        spikes = simulate_population(cur)
        R = ft.population_corr_matrix(spikes, window=int(WINDOW_MS / ft.POP_BIN_MS))
        rhos.append(ft.offdiag_mean(R))
        rates.append(spikes.mean() * 1000.0 / ft.POP_BIN_MS)
    rhos = np.array(rhos)
    return rhos.mean(), rhos.std(ddof=1) / np.sqrt(N_RUNS), float(np.mean(rates))


def sweep(params, kind, rng):
    shape = (len(params), len(C_VALUES))
    rho, sem, rate = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for i, p in enumerate(params):
        t0 = time.time()
        for j, c in enumerate(C_VALUES):
            mu, var = (p, MU_SWEEP_VAR) if kind == "mu" else (0.0, p)
            rho[i, j], sem[i, j], rate[i, j] = run_condition(mu, var, c, rng)
        print(f"  {kind}={p:g}: rate {rate[i].mean():.1f} Hz, "
              f"<rho>(c=0.5)={rho[i, 5]:.3f}  ({time.time() - t0:.0f} s)")
    return rho, sem, rate


def simulate_all():
    rng = np.random.default_rng(SEED)
    print("mu sweep (sigma^2 = %g pA^2)" % MU_SWEEP_VAR)
    mu_rho, mu_sem, mu_rate = sweep(MU_VALUES, "mu", rng)
    print("sigma^2 sweep (mu = 0)")
    var_rho, var_sem, var_rate = sweep(VAR_VALUES, "var", rng)
    data = dict(c=C_VALUES, mu=MU_VALUES, var=VAR_VALUES,
                mu_rho=mu_rho, mu_sem=mu_sem, mu_rate=mu_rate,
                var_rho=var_rho, var_sem=var_sem, var_rate=var_rate,
                window_ms=WINDOW_MS, n_runs=N_RUNS, seconds=SECONDS,
                n_neurons=N_NEURONS, mu_sweep_var=MU_SWEEP_VAR)
    np.savez(CACHE, **data)
    return data


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def ramp(n, cmap="Blues", lo=0.35, hi=0.95):
    return [matplotlib.colormaps[cmap](x) for x in np.linspace(lo, hi, n)]


def line_panel(ax, c, rho, sem, rate, params, fmt, pyr):
    ax.plot([0, 0.6], [0, 0.6], ls="--", lw=1, color=MUTED, zorder=0)
    ax.text(0.285, 0.305, "ρ = c", color=MUTED, rotation=62, fontsize=9, clip_on=True)
    for k, col in enumerate(ramp(len(params))):
        ax.fill_between(c, rho[k] - sem[k], rho[k] + sem[k], color=col,
                        alpha=0.2, lw=0)
        ax.plot(c, rho[k], color=col, lw=2, marker="o", ms=6, mec="white",
                mew=1, label=f"{fmt(params[k])}  ({rate[k].mean():.0f} Hz)")
    pc, pm, ps, pn, pr = pyr
    keep = pc <= 0.6 + 1e-9
    ax.errorbar(pc[keep], pm[keep], yerr=ps[keep], color="black", lw=2,
                ls="--", marker="s", ms=7, mfc="black", capsize=3,
                label=f"Real PYR ({pn[0]} pairs, {pr.mean():.0f} Hz)", zorder=5)
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(-0.02, 0.34)
    ax.set_xlabel("Input correlation  c")
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)


def contour_panel(fig, ax, c, params, rho, ylabel, log=False):
    levels = np.linspace(0, max(0.3, np.ceil(rho.max() * 20) / 20), 13)
    cf = ax.contourf(c, params, rho, levels=levels, cmap="Blues")
    cs = ax.contour(c, params, rho, levels=levels[2::2], colors="white", linewidths=0.8)
    ax.clabel(cs, fmt="%.2f", fontsize=8, colors=INK)
    if log:
        ax.set_yscale("log")
        ax.set_yticks(params)
        ax.set_yticklabels([f"{p:g}" for p in params])
        ax.minorticks_off()
    ax.set_xlabel("Input correlation  c")
    ax.set_ylabel(ylabel)
    fig.colorbar(cf, ax=ax, label="⟨ρ⟩")


def plot(d, pyr):
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, ax = plt.subplots(2, 2, figsize=(13, 10.5),
                           gridspec_kw={"height_ratios": [1.15, 1]})
    c = d["c"]
    line_panel(ax[0, 0], c, d["mu_rho"], d["mu_sem"], d["mu_rate"], d["mu"],
               lambda p: f"μ = {p:+g} pA", pyr)
    line_panel(ax[0, 1], c, d["var_rho"], d["var_sem"], d["var_rate"], d["var"],
               lambda p: f"σ² = {p:g} pA²", pyr)
    ax[0, 0].set_ylabel(f"Mean pairwise output correlation ⟨ρ⟩\n"
                        f"(average over 190 pairs, T = {int(d['window_ms'])} ms)")
    ax[0, 0].set_title(f"A   FT: vary input mean μ  (σ² = {d['mu_sweep_var']:g} pA²)",
                       loc="left", color=INK)
    ax[0, 1].set_title("B   FT: vary input variance σ²  (μ = 0)",
                       loc="left", color=INK)
    contour_panel(fig, ax[1, 0], c, d["mu"], d["mu_rho"], "Input mean μ (pA)")
    contour_panel(fig, ax[1, 1], c, d["var"], d["var_rho"],
                  "Input variance σ² (pA², log scale)", log=True)
    ax[1, 0].set_title("C   ⟨ρ⟩ over (c, μ)", loc="left", color=INK)
    ax[1, 1].set_title("D   ⟨ρ⟩ over (c, σ²)", loc="left", color=INK)
    fig.suptitle("Hong et al. (2012) Fig 4A replication: FT population model "
                 "with real L5 pyramidal cells as reference",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.text(0.01, 0.005,
             f"FT: {int(d['n_neurons'])} neurons, {int(d['n_runs'])} runs × "
             f"{d['seconds']:g} s per point, OU τ = {ft.TAU_MS:g} ms; shaded = ±1 SEM "
             "across runs. Rates in legends are FT output rates.\n"
             "Real PYR: Linaro et al. 2019 (figshare 7241144), OU current clamp, "
             "100 × 1 s trials per c, one operating point per cell; error bars = "
             "±1 SEM across pairs.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(OUT_PNG, dpi=200)
    print("saved", OUT_PNG)


def print_table(d):
    for kind, params, unit in (("mu", d["mu"], "pA"), ("var", d["var"], "pA^2")):
        print(f"\n<rho> by {kind} ({unit}) — columns c = " +
              " ".join(f"{c:g}" for c in d["c"]))
        for k, p in enumerate(params):
            print(f"{p:8g} ({d[kind + '_rate'][k].mean():5.1f} Hz): " +
                  " ".join(f"{v:6.3f}" for v in d[kind + "_rho"][k]))


def main():
    if CACHE.exists() and "--rerun" not in sys.argv:
        d = dict(np.load(CACHE))
        print("loaded cache", CACHE)
    else:
        d = simulate_all()
    pyr = rho_vs_c([cl for cl in load_dataset().values() if cl.cell_type == "PYR"],
                   window_ms=WINDOW_MS)
    print_table(d)
    print("\nReal PYR <rho>: " + " ".join(f"c={c:g}:{m:.3f}" for c, m in zip(pyr[0], pyr[1])))
    plot(d, pyr)


if __name__ == "__main__":
    main()
