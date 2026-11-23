"""Gaussian paired-current experiments and Hong-style spike statistics."""
import numpy as np
from scipy import signal
from scipy.fft import irfft, next_fast_len, rfft

from model import FS, simulate

LAGS = np.arange(-200, 201)
WINDOWS = (200, 40)
TARGETS = np.array([6, 12, 20])
LEVELS = np.arange(6) / 10
BETAS = (0, -21)
PAIRS = ("I–I", "D–D", "I–D")
SD = 150.
BURN = 500
DURATION = 16000
N = 8
SEEDS = np.arange(24) + 100
TRAIN = 12


def rate_pilot():
    means = (np.arange(150, 281, 10), np.arange(330, 551, 10))
    params = [(beta, m) for beta, grid in zip(BETAS, means) for m in grid]
    measured = []
    for seed in [21, 22, 23, 24]:
        white = np.random.default_rng(seed).normal(size=(12000 + BURN) * FS)
        spikes, _ = simulate([white] * len(params), [b for b, m in params],
                             means=[m for b, m in params], sds=[SD] * len(params))
        measured.append([np.count_nonzero(x >= BURN) / 12 for x in spikes])
    rates = np.mean(measured, axis=0)
    chosen = []
    for grid, curve in zip(means, np.split(rates, [len(means[0])])):
        if curve[0] > TARGETS[0] or curve[-1] < TARGETS[-1]:
            raise ValueError(f"Pilot grid does not bracket target rates: {curve}")
        chosen.append(np.interp(TARGETS, np.maximum.accumulate(curve), grid))
    return dict(means=np.array(chosen), grid=np.array(params), pilot_rates=rates)


def binned(spikes, duration):
    return np.array([np.bincount(s.astype(int), minlength=duration)[:duration] for s in spikes])


def lag_slice(spectrum, nfft, n):
    values = irfft(spectrum, nfft)[LAGS % nfft]
    return values / (n - np.abs(LAGS))


def summarize(trials):
    """Within-trial minus between-trial CCG; positive lag means the second group follows."""
    a = (trials[0].astype(float) - trials[1]) / np.sqrt(2)
    n = a.shape[-1]
    nfft = next_fast_len(n + 200, real=True)
    fourier = rfft(a, nfft)
    autos = [lag_slice((abs(f) ** 2).mean(axis=0), nfft, n) for f in fourier]
    ccgs, rho = [], []
    for ai, bi in [(0, 0), (1, 1), (0, 1)]:
        fa, fb = fourier[ai], fourier[bi]
        # Same-index cells replay the same private drive across modes; exclude those pairs.
        cross = fa.sum(axis=0).conj() * fb.sum(axis=0) - (fa.conj() * fb).sum(axis=0)
        ccg = lag_slice(cross / (len(fa) * (len(fa) - 1)), nfft, n)
        ccgs.append(ccg)
        row = []
        for window in WINDOWS:
            w = np.maximum(window - abs(LAGS), 0)
            row.append((w @ ccg) / np.sqrt((w @ autos[ai]) * (w @ autos[bi])))
        rho.append(row)
    return np.array(ccgs), np.array(rho)


def sta_sum(white, times):
    past = 200 * FS
    idx = (times * FS).astype(int)
    idx = idx[(idx >= past) & (idx < len(white))]
    offsets = np.arange(-past, 1)
    return (white[idx[:, None] + offsets] - white.mean()).sum(axis=0), len(idx)


def predicted_ccg(sa, sb, ra, rb, c):
    # STAs run from the past to spike time. Integrate over both 1-ms output bins.
    fine = signal.correlate(sa, sb, mode="full")
    weights = np.maximum(FS - abs(np.arange(-FS + 1, FS)), 0) / FS**2
    smoothed = np.convolve(fine, weights, mode="same")
    return c * ra * rb * 1e-6 * smoothed[len(sa) - 1 + FS * LAGS]


def gaussian(seed, level, means):
    n = (DURATION + BURN) * FS
    trials, currents, example = [], [], {}
    sta = np.zeros((2, 200 * FS + 1))
    nsta = np.zeros(2, dtype=int)
    spike_times = []
    for repeat in range(2):
        rng = np.random.default_rng([seed, repeat])
        drive = np.sqrt(level) * rng.normal(size=n) + np.sqrt(1 - level) * rng.normal(size=(N, n))
        record = (0, 1)
        spikes, traces = simulate(list(drive) * 2, np.repeat(BETAS, N),
                                  means=np.repeat(means, N), sds=[SD] * (2*N), record=record)
        currents.append(np.corrcoef(traces[0][1][BURN*FS:], traces[1][1][BURN*FS:])[0, 1])
        spikes = [x[(x >= BURN) & (x < BURN + DURATION)] - BURN for x in spikes]
        spike_times.extend(spikes)
        if repeat == 0:
            example['input'] = np.array([traces[k][1][BURN*FS:(BURN+1000)*FS] for k in record])
        if level == 0 and seed < SEEDS[TRAIN]:
            for k, times in enumerate(spikes):
                acc, num = sta_sum(drive[k % N, BURN*FS:], times)
                sta[k // N] += acc
                nsta[k // N] += num
        trials.append(binned(spikes, DURATION).reshape(2, N, -1))
    counts = np.array(trials)
    ccg, rho = summarize(counts)
    rates = counts.sum(axis=-1).mean(axis=(0,2)) * 1000 / DURATION
    offsets = np.r_[0, np.cumsum([len(s) for s in spike_times])]
    return dict(ccg=ccg, rho=rho, rate=rates, actual_c=np.mean(currents), sta=sta,
                nsta=nsta, spikes=np.concatenate(spike_times),
                offsets=offsets, **example)
