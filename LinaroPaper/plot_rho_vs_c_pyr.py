"""
Mean pairwise output correlation <rho> vs input correlation c:
real L5 pyramidal cells (Linaro et al. 2019, OU current clamp) vs FT model.
<rho> = average over all neuron pairs of the spike-count correlation
(15 pairs for 6 PYR cells, 190 pairs for 20 FT neurons). T = 40 ms.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from linaro_loader import (load_dataset, population_spikes,
                           population_corr_matrix, sliding_counts)

WINDOW_MS = 40
HERE = Path(__file__).resolve().parent             # LinaroPaper/
FT_FILE = HERE.parent / "ft_population_dataset.npz"  # project root
OUT_PNG = HERE / "rho_vs_c_PYR_mean.png"
C_FT, C_PYR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def pyr_pair_rhos(window_ms=WINDOW_MS):
    """Return c values, (n_c, n_pairs) rho array, pair labels, mean rates."""
    pyr = [cl for cl in load_dataset().values() if cl.cell_type == "PYR"]
    cs = sorted({c for cl in pyr for c in cl.c_values})
    iu = np.triu_indices(len(pyr), 1)
    rhos, rates = [], []
    for c in cs:
        spikes, ids = population_spikes(pyr, c)
        assert len(ids) == len(pyr)
        rhos.append(population_corr_matrix(spikes, window_ms)[iu])
        rates.append(np.mean([cl.get(c).nu for cl in pyr]))
    labels = [f"{pyr[i].cell_id[-3:]}–{pyr[j].cell_id[-3:]}" for i, j in zip(*iu)]
    return np.array(cs), np.array(rhos), labels, np.array(rates)


def ft_pair_rhos(window_ms=WINDOW_MS, path=FT_FILE):
    """Per-pair rho for FT, averaged over the runs at each c: (n_c, 190)."""
    d = np.load(path)
    spikes, c = d["spikes"], d["input_corr"]
    w = int(round(window_ms / float(d["bin_ms"])))
    iu = np.triu_indices(spikes.shape[1], 1)
    per_run = np.array([np.corrcoef(sliding_counts(s, w))[iu] for s in spikes])
    cs = np.unique(c)
    return cs, np.array([per_run[c == v].mean(axis=0) for v in cs]), \
        float(d["firing_rates_hz"].mean())


def style(ax):
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, zorder=0)
    ax.text(0.70, 0.745, "ρ = c", color=MUTED, rotation=36, fontsize=9)
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.08, 1.0)
    ax.set_xlabel("Input correlation  c")
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    c_p, r_p, labels, rates_p = pyr_pair_rhos()
    c_f, r_f, rate_f = ft_pair_rhos()
    n_p, n_f = r_p.shape[1], r_f.shape[1]

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for cs, r, col, mk, name in (
            (c_p, r_p, C_PYR, "s",
             f"Real PYR cells: mean over {n_p} pairs (6 cells, {rates_p.mean():.0f} Hz)"),
            (c_f, r_f, C_FT, "o",
             f"FT model: mean over {n_f} pairs (20 neurons, {rate_f:.0f} Hz)")):
        m = r.mean(axis=1)
        se = r.std(axis=1, ddof=1) / np.sqrt(r.shape[1])
        ax.fill_between(cs, m - se, m + se, color=col, alpha=0.2, lw=0, zorder=2)
        ax.plot(cs, m, color=col, lw=2.5, marker=mk, ms=8, mec="white",
                mew=1.5, zorder=3, label=name)
        ax.text(cs[-1] + 0.025, m[-1], f"{m[-1]:.2f}", color=INK,
                va="center", fontsize=10)
    style(ax)
    ax.set_xlim(-0.04, 1.1)
    ax.set_ylabel(f"Mean pairwise output correlation  ⟨ρ⟩\n"
                  f"(average of ρ over all neuron pairs, T = {WINDOW_MS} ms)")
    ax.set_title("Mean pairwise output correlation vs input correlation",
                 loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.text(0.01, 0.01, "⟨ρ⟩ = average of the spike-count correlation over every "
             "neuron pair; shaded = ±1 SEM across pairs.\nReal data: Linaro et al. "
             "2019 (figshare 7241144), L5 pyramidal cells, OU current clamp, "
             "100 × 1 s trials per c.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_PNG, dpi=200)
    print("saved", OUT_PNG)
    for name, cs, r in (("PYR", c_p, r_p), ("FT", c_f, r_f)):
        print(name, " ".join(f"c={c:g}:{v:.3f}" for c, v in zip(cs, r.mean(axis=1))))


if __name__ == "__main__":
    main()
