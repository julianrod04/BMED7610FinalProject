"""
Output correlation rho vs input correlation c:
FT model population (ft_population_dataset.npz) vs real L5 neurons
(Linaro et al. 2019, data/variable_c_cells).

Both use the same count window (T = 40 ms) and the mean over all pairs.
Real data: trial-shuffle-corrected count correlation (linaro_loader).
FT: np.corrcoef of sliding counts over each 11.8 s run (as FT_model.py),
    averaged over the 12 runs at each c.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from linaro_loader import load_dataset, rho_vs_c, sliding_counts, offdiag_mean

WINDOW_MS = 40
HERE = Path(__file__).resolve().parent             # LinaroPaper/
FT_FILE = HERE.parent / "ft_population_dataset.npz"  # project root
OUT_PNG = HERE / "rho_vs_c_real_vs_FT.png"

# fixed categorical order (validated default palette, slots 1-4)
COLORS = {"FT model": "#2a78d6", "PYR": "#eb6834", "PV": "#1baf7a", "SOM": "#eda100"}
MARKERS = {"FT model": "o", "PYR": "s", "PV": "^", "SOM": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def ft_rho_vs_c(path=FT_FILE, window_ms=WINDOW_MS):
    d = np.load(path)
    spikes, c = d["spikes"], d["input_corr"]      # (runs, neurons, 1 ms bins)
    w = int(round(window_ms / float(d["bin_ms"])))
    per_run = np.array([offdiag_mean(np.corrcoef(sliding_counts(s, w)))
                        for s in spikes])
    cs = np.unique(c)
    m = np.array([per_run[c == v].mean() for v in cs])
    se = np.array([per_run[c == v].std(ddof=1) / np.sqrt((c == v).sum()) for v in cs])
    rate = float(d["firing_rates_hz"].mean())
    n = spikes.shape[1]
    return cs, m, se, rate, n


def main():
    series = {}
    cs, m, se, rate, n = ft_rho_vs_c()
    series["FT model"] = (cs, m, se, f"FT model ({n} neurons, {rate:.0f} Hz)")

    cells = load_dataset()
    for ctype in ("PYR", "PV", "SOM"):
        group = [cl for cl in cells.values() if cl.cell_type == ctype]
        c, mm, ss, npairs, rates = rho_vs_c(group, window_ms=WINDOW_MS)
        series[ctype] = (c, mm, ss, f"{ctype} real ({len(group)} cells, "
                                    f"{rates.mean():.0f} Hz)")

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, zorder=1)
    ax.text(0.62, 0.66, "ρ = c", color=MUTED, rotation=35, fontsize=10)

    for name, (c, mm, ss, label) in series.items():
        col = COLORS[name]
        ax.fill_between(c, mm - ss, mm + ss, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(c, mm, color=col, lw=2, marker=MARKERS[name], ms=8,
                mec="white", mew=1.5, label=label, zorder=3)

    # direct labels at line ends (nudged apart vertically)
    ends = sorted(((mm[-1], c[-1], name) for name, (c, mm, _, _) in series.items()))
    last_y = -1
    for y, x, name in ends:
        y_lab = max(y, last_y + 0.035)
        ax.text(x + 0.025, y_lab, name, color=INK, va="center", fontsize=10)
        last_y = y_lab

    ax.set_xlim(-0.02, 1.12)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("Input correlation  c")
    ax.set_ylabel(f"Output correlation  ρ  (count window T = {WINDOW_MS} ms)")
    ax.set_title("Output vs input correlation: real L5 neurons vs FT model",
                 color=INK, loc="left")
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.text(0.01, 0.01, "Shaded: ±1 SEM (real: across pairs; FT: across 12 runs). "
             "Real data: Linaro et al. 2019 (figshare 7241144).",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_PNG, dpi=200)
    print("saved", OUT_PNG)
    for name, (c, mm, ss, label) in series.items():
        print(label, " ".join(f"{a:g}:{b:.3f}" for a, b in zip(c, mm)))


if __name__ == "__main__":
    main()
