"""
Loader for the Linaro, Ocker, Doiron & Giugliano (2019) "variable_c_cells" dataset.

Source: figshare DOI 10.6084/m9.figshare.7241144 (CC BY 4.0)
Paper:  "Correlation transfer by layer 5 cortical neurons under recreated
         synaptic inputs in vitro", J Neurosci 39:7648 (2019).

Layout on disk
--------------
data/variable_c_cells/<CELL_ID>/CELL_TYPE                 -> "PYR", "PV" or "SOM"
data/variable_c_cells/<CELL_ID>/correlations/<CC>/correlations.mat

<CC> encodes the input correlation c: "00"=0, "015"=0.15, "03"=0.3, "10"=1.0.
A "_bis" suffix is a second block at the same c (only 20150419A01/03_bis).

How the experiment works (from the authors' correlations.py / MATLAB scripts)
---------------------------------------------------------------------------
Each cell was recorded alone, with 100 trials per value of c. On trial k the
input was   private_noise(random seed)*sqrt(1-c) + common_noise(seed_k)*sqrt(c).
The common seeds seed_k come from a fixed RNG seed (5061983), so trial k has
the SAME common noise in every cell and every session. A "pair" (or a whole
population) is therefore built *virtually*: trial k of cell A goes with trial
k of cell B. The authors correct for stimulus locking by subtracting the
trial-shuffled product (trial k of A with trial k+1 of B), which has no shared
input. This is the same idea as the shuffle predictor in the Hong et al. CCG.

Stimulus
- PYR cells (2013 recordings): current clamp, OU current (de la Rocha 2007
  protocol). c in {0, .2, .4, .6, .8, 1}.
- PV and SOM interneurons (2015 recordings): dynamic clamp with OU
  conductances (exc. tau=5 ms, E=0 mV; inh. tau=10 ms, E=-80 mV).
  c in {0, .15, .3, .45, .6, .8, 1}.
The stimulus waveforms themselves are NOT in the files (only spikes), so
STA/STC cannot be computed from this dataset.

Spike times are in seconds, relative to the start of the analysed window:
stimulus onset + 0.1 s transient (ttran) removed, 1.0 s long (dur - ttran).

Quick use
---------
    from linaro_loader import load_dataset, population_spikes, pair_rho
    cells = load_dataset()                       # dict cell_id -> Cell
    pyr = [c for c in cells.values() if c.cell_type == "PYR"]
    rho = pair_rho(pyr[0], pyr[1], c=0.6)       # one virtual pair
    spikes, ids = population_spikes(pyr, c=0.6) # (trials, n_cells, bins) like FT
    R = population_corr_matrix(spikes)          # shuffle-corrected rho matrix
"""

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

PROJECT_DIR = Path(__file__).resolve().parent.parent  # BMED7610FinalProject/
DEFAULT_ROOT = PROJECT_DIR / "data" / "variable_c_cells"
TRIAL_DURATION_S = 1.0  # analysed window per trial (dur - ttran)


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------
@dataclass
class Condition:
    """One block of trials of one cell at one input correlation c."""
    c: float
    block: int                    # 0 for the main block, 1 for "_bis"
    spiketimes: list              # list (n_trials) of 1-D arrays, seconds
    rate_hz: np.ndarray           # per-trial firing rate (Hz)
    nu: float                     # mean firing rate (Hz)
    cv: float                     # ISI coefficient of variation
    meta: dict = field(default_factory=dict)  # ttran, t0, dur, tau, R, C, ...

    @property
    def n_trials(self):
        return len(self.spiketimes)


@dataclass
class Cell:
    cell_id: str
    cell_type: str                # "PYR", "PV" or "SOM"
    mode: str                     # "current" (OU current) or "conductance"
    conditions: list              # list of Condition
    membrane: dict                # tau (s), R (Ohm), C (F), tau_eff (s)

    @property
    def c_values(self):
        return sorted({cond.c for cond in self.conditions})

    def get(self, c, block=0):
        """Condition at input correlation c (tolerant float match)."""
        for cond in self.conditions:
            if np.isclose(cond.c, c) and cond.block == block:
                return cond
        raise KeyError(f"{self.cell_id}: no block {block} at c={c}")

    def has(self, c, block=0):
        return any(np.isclose(k.c, c) and k.block == block
                   for k in self.conditions)

    def __repr__(self):
        cs = ", ".join(f"{c:g}" for c in self.c_values)
        return (f"Cell({self.cell_id}, {self.cell_type}, {self.mode}, "
                f"c=[{cs}])")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def parse_c(dirname):
    """'00'->0, '015'->0.15, '03'->0.3, '10'->1.0, '03_bis'->(0.3, block 1)."""
    m = re.fullmatch(r"(\d+)(_bis)?", dirname)
    if m is None:
        raise ValueError(f"unrecognised correlation folder: {dirname}")
    digits = m.group(1)
    c = int(digits) / 10 ** (len(digits) - 1)
    return c, (1 if m.group(2) else 0)


def _scalar(m, key, default=np.nan):
    return float(m[key]) if key in m else default


def load_condition(mat_path, c, block):
    m = sio.loadmat(str(mat_path), squeeze_me=True)
    st = [np.atleast_1d(np.asarray(s, dtype=float)) for s in m["spiketimes"]]
    rates = m["rates"]
    rates = rates.toarray().ravel() if sp.issparse(rates) else np.asarray(rates)
    meta = {k: _scalar(m, k) for k in
            ("ttran", "t0", "dur", "binWidth", "windowSize", "T",
             "tau", "R", "C", "tau_eff", "Ee", "Ei")}
    return Condition(c=c, block=block, spiketimes=st,
                     rate_hz=rates.astype(float), nu=_scalar(m, "nu"),
                     cv=_scalar(m, "cv"), meta=meta)


def load_cell(cell_dir):
    cell_dir = Path(cell_dir)
    cell_type = (cell_dir / "CELL_TYPE").read_text().strip()
    conds = []
    for d in sorted((cell_dir / "correlations").iterdir()):
        f = d / "correlations.mat"
        if not f.exists() or (d / "DO_NOT_USE").exists():
            continue
        c, block = parse_c(d.name)
        conds.append(load_condition(f, c, block))
    conds.sort(key=lambda k: (k.c, k.block))
    mode = "current" if np.isnan(conds[0].meta["Ee"]) else "conductance"
    # Membrane properties: median across conditions (they are re-estimated
    # per block and are very similar).
    membrane = {k: float(np.nanmedian([cd.meta[k] for cd in conds]))
                for k in ("tau", "R", "C", "tau_eff")}
    return Cell(cell_dir.name, cell_type, mode, conds, membrane)


def load_dataset(root=DEFAULT_ROOT):
    """Return {cell_id: Cell} for every cell folder under root."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found - unzip variable_c_cells.zip there")
    return {d.name: load_cell(d) for d in sorted(root.iterdir())
            if (d / "CELL_TYPE").exists()}


def summary_table(cells):
    """Print one line per cell: type, mode, membrane tau, rate at each c."""
    all_c = sorted({c for cell in cells.values() for c in cell.c_values})
    print(f"{'cell':12s} {'type':4s} {'mode':11s} {'tau_m':>6s}  " +
          " ".join(f"c={c:<4g}" for c in all_c))
    for cell in cells.values():
        rates = [f"{cell.get(c).nu:6.1f}" if cell.has(c) else "     -"
                 for c in all_c]
        print(f"{cell.cell_id:12s} {cell.cell_type:4s} {cell.mode:11s} "
              f"{cell.membrane['tau'] * 1e3:5.1f}ms  " + " ".join(rates))


# ----------------------------------------------------------------------------
# Binning (same conventions as FT_model.py: 1 ms bins, uint8 counts)
# ----------------------------------------------------------------------------
def binarize(spiketimes, bin_ms=1.0, duration_s=TRIAL_DURATION_S):
    """list of spike-time arrays -> (n_trials, n_bins) uint8 count array."""
    n_bins = int(round(duration_s * 1e3 / bin_ms))
    out = np.zeros((len(spiketimes), n_bins), dtype=np.uint8)
    for k, s in enumerate(spiketimes):
        idx = np.floor(s * 1e3 / bin_ms).astype(int)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        np.add.at(out[k], idx, 1)
    return out


def _check_same_mode(cells):
    """PYR (OU current) and PV/SOM (OU conductance) got DIFFERENT common
    noise signals: their cross pairs have rho ~ 0 even at c = 1 (checked).
    PV x SOM pairs are fine (same conductance seeds)."""
    modes = {cl.mode for cl in cells}
    if len(modes) > 1:
        raise ValueError("cannot pair current-clamp (PYR) with conductance-"
                         "clamp (PV/SOM) cells: their common inputs differ")


def population_spikes(cells, c, bin_ms=1.0, block=0):
    """
    Build a virtual population at input correlation c.

    Trial k of every cell received the same common input, so stacking cells
    along axis 1 gives the same layout as FT_model.py's `spikes` array:
    (n_trials, n_cells, n_bins). Cells without data at c are skipped.
    Returns (spikes, cell_ids).
    """
    cells = [cl for cl in cells if cl.has(c, block)]
    if not cells:
        raise ValueError(f"no cells have data at c={c}")
    _check_same_mode(cells)
    n_trials = min(cl.get(c, block).n_trials for cl in cells)
    stacks = [binarize(cl.get(c, block).spiketimes[:n_trials], bin_ms)
              for cl in cells]
    return np.stack(stacks, axis=1), [cl.cell_id for cl in cells]


# ----------------------------------------------------------------------------
# Correlation measures (authors' method: shuffle-corrected count covariance)
# ----------------------------------------------------------------------------
def sliding_counts(binned, window_bins):
    """(..., n_bins) -> (..., n_bins - window + 1) spike counts in a sliding
    window (step = 1 bin), as in binarizeSpikeTrain.m."""
    cs = np.cumsum(binned, axis=-1, dtype=np.float64)
    cs = np.concatenate([np.zeros(cs.shape[:-1] + (1,)), cs], axis=-1)
    return cs[..., window_bins:] - cs[..., :-window_bins]


def shuffle_corrected_cov(ni, nj):
    """
    ni, nj: (n_trials, n_windows) sliding counts of two cells, trial-aligned.
    cov = < n_i(k) n_j(k) - n_i(k) n_j(k+1) >  (trial k+1 wraps to trial 0),
    i.e. computeCorrelationCoefficient.m. Works for i == j (variance).
    """
    nj_shift = np.roll(nj, -1, axis=0)
    return float(np.mean(ni * nj - ni * nj_shift))


def pair_rho(cell_i, cell_j, c, window_ms=40.0, bin_ms=1.0,
             block_i=0, block_j=None):
    """
    Spike-count correlation coefficient of a virtual pair at input corr c.
    window_ms=40 is the authors' default. Trials are only 1 s long, so a
    200 ms window (Hong et al. / FT_model.py) gives a much noisier estimate.
    For a same-cell pair use cell_j=cell_i, block_i=0, block_j=1.
    """
    if block_j is None:
        block_j = block_i
    _check_same_mode([cell_i, cell_j])
    ci, cj = cell_i.get(c, block_i), cell_j.get(c, block_j)
    n = min(ci.n_trials, cj.n_trials)
    w = int(round(window_ms / bin_ms))
    ni = sliding_counts(binarize(ci.spiketimes[:n], bin_ms), w)
    nj = sliding_counts(binarize(cj.spiketimes[:n], bin_ms), w)
    cov = shuffle_corrected_cov(ni, nj)
    var = np.sqrt(shuffle_corrected_cov(ni, ni) * shuffle_corrected_cov(nj, nj))
    return cov / var if var > 0 else np.nan


def population_corr_matrix(spikes, window_ms=40.0, bin_ms=1.0):
    """
    spikes: (n_trials, n_cells, n_bins) from population_spikes().
    Returns the (n_cells, n_cells) shuffle-corrected rho matrix (diag = 1).
    Same role as FT_model.population_corr_matrix, but trial-based.
    """
    w = int(round(window_ms / bin_ms))
    n = sliding_counts(spikes, w)                     # (K, N, W)
    n_shift = np.roll(n, -1, axis=0)
    K, N, W = n.shape
    raw = np.einsum("kiw,kjw->ij", n, n) / (K * W)
    shuf = np.einsum("kiw,kjw->ij", n, n_shift) / (K * W)
    cov = raw - 0.5 * (shuf + shuf.T)                 # symmetric estimate
    d = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        R = cov / np.outer(d, d)
    R[d == 0, :] = np.nan
    R[:, d == 0] = np.nan
    return R


def population_ccg(spikes, maxlag_ms=40, bin_ms=1.0):
    """
    Shuffle-corrected CCG averaged over all distinct cell pairs, in Hz^2
    (same units as FT_model.get_shift_ccg). spikes: (trials, cells, bins).
    """
    from scipy.signal import correlate
    x = spikes.astype(np.float64)
    x_shift = np.roll(x, -1, axis=0)
    K, N, B = x.shape
    L = int(round(maxlag_ms / bin_ms))
    lags = np.arange(-L, L + 1)
    center = B - 1
    acc = np.zeros(lags.size)
    npairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            same = sum(correlate(x[k, i], x[k, j], "full", method="fft")
                       for k in range(K))
            shuf = sum(correlate(x[k, i], x_shift[k, j], "full", method="fft")
                       for k in range(K))
            acc += (same - shuf)[center + lags] / K
            npairs += 1
    ccg = acc / max(npairs, 1) / (B - np.abs(lags))
    return lags * bin_ms, ccg / (bin_ms * 1e-3) ** 2


def offdiag_mean(matrix):
    idx = np.triu_indices(matrix.shape[0], 1)
    v = matrix[idx]
    return float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan


def rho_vs_c(cells, window_ms=40.0, bin_ms=1.0):
    """
    Mean pairwise rho (over all virtual pairs among `cells`) at every c that
    at least two of them share. Returns (c_values, mean_rho, sem_rho, n_pairs,
    mean_rate_hz).
    """
    all_c = sorted({c for cl in cells for c in cl.c_values})
    cs, mean, sem, npairs, rates = [], [], [], [], []
    for c in all_c:
        group = [cl for cl in cells if cl.has(c)]
        if len(group) < 2:
            continue
        spikes, _ = population_spikes(group, c, bin_ms)
        R = population_corr_matrix(spikes, window_ms, bin_ms)
        v = R[np.triu_indices(len(group), 1)]
        v = v[np.isfinite(v)]
        cs.append(c)
        mean.append(v.mean())
        sem.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan)
        npairs.append(v.size)
        rates.append(np.mean([cl.get(c).nu for cl in group]))
    return (np.array(cs), np.array(mean), np.array(sem), np.array(npairs),
            np.array(rates))


if __name__ == "__main__":
    cells = load_dataset()
    summary_table(cells)
    for ctype in ("PYR", "PV", "SOM"):
        group = [cl for cl in cells.values() if cl.cell_type == ctype]
        c, m, s, n, r = rho_vs_c(group)
        print(f"\n{ctype} ({len(group)} cells): mean pairwise rho (T=40 ms)")
        for row in zip(c, m, s, n, r):
            print("  c=%.2f  rho=%.3f +- %.3f  (%d pairs, %.1f Hz)" % row)
