"""Fixed-cell stimulus controls, the single-cell demonstration, and the hybrid experiment."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from scipy.signal import lfilter

from model import FS, simulate
from transfer import BETAS, BURN, N, binned, summarize

HYBRID_BETA = -13
HYBRID_SD = 40
HYBRID_MS = 32000
EVENT_HZ = 1
HYBRID_SEEDS = np.arange(16) + 301
CONDITIONS = ("Slow shared", "Events shared", "Both shared", "Events made private")
SHARES = ((.75, 0), (0, .75), (.375, .375), (.375, 0))


def ou(rng, n, tau):
    a = np.exp(-1 / (FS * tau))
    return lfilter([np.sqrt(1 - a*a)], [1, -a], rng.normal(size=n),
                   zi=[a*rng.normal()])[0]


def events(rng, n):
    count = rng.poisson(EVENT_HZ * n / (1000*FS))
    return rng.integers(n, size=count), rng.uniform(size=count)


def shot_noise(indices, n):
    fall, rise = np.exp(-1/(FS*3)), np.exp(-1/(FS*.5))
    impulses = np.bincount(indices, minlength=n)
    pulse = lfilter([1.], [1., -fall], impulses) - lfilter([1.], [1., -rise], impulses)
    lam = EVENT_HZ / (1000*FS)
    mean = lam * (1/(1-fall) - 1/(1-rise))
    variance = lam * (1/(1-fall**2) + 1/(1-rise**2) - 2/(1-fall*rise))
    return (pulse - mean) / np.sqrt(variance)


def hybrid_inputs(seed, repeat, condition, duration, cells):
    # Every cell has the same marginal variance budget: 40% slow, 40% pulses, 20% background.
    n = (duration + BURN)*FS
    slow_share, event_share = SHARES[condition]
    common_rng = np.random.default_rng([seed, repeat, 0])
    slow = ou(common_rng, n, 100)
    candidates, marks = events(common_rng, n)
    shared = candidates[marks < event_share]
    # The same event times serve as sham triggers in the private-event control.
    triggers = candidates[marks < (.375 if condition == 3 else event_share)]
    currents, all_events = [], []
    for k in range(cells):
        rng = np.random.default_rng([seed, repeat, k+1])
        private_slow = ou(rng, n, 100)
        private, marks = events(rng, n)
        arrivals = np.sort(np.r_[shared, private[marks >= event_share]])
        slow_i = np.sqrt(slow_share)*slow + np.sqrt(1-slow_share)*private_slow
        currents.append(HYBRID_SD * (np.sqrt(.4)*slow_i + np.sqrt(.4)*shot_noise(arrivals, n)
                                       + np.sqrt(.2)*ou(rng, n, 5)))
        all_events.append(arrivals)
    return np.array(currents), slow, np.sort(triggers), all_events


def hybrid_pilot():
    grid = np.arange(315., 401., 5)
    rates = []
    for seed in (40, 41, 42):
        drive = hybrid_inputs(seed, 0, 2, 20000, 1)[0][0]
        spikes, _ = simulate([drive+m for m in grid], [HYBRID_BETA]*len(grid))
        rates.append([np.count_nonzero(x >= BURN)/20 for x in spikes])
    rate = np.mean(rates, axis=0)
    if not rate.min() < 12 < rate.max():
        raise ValueError("Hybrid pilot failed to bracket 12 Hz")
    return np.interp(12, np.maximum.accumulate(rate), grid), grid, rate


def hybrid_trial(seed, condition, mean):
    trains, stim_corr, stim_sd = [], [], []
    by_slow, exposure = np.zeros(5), np.zeros(5)
    event_pairs, n_events = np.zeros(41), 0
    example = {}
    for rep in range(2):
        current, slow, triggers, events_i = hybrid_inputs(seed, rep, condition, HYBRID_MS, N)
        current += mean
        spikes, _ = simulate(current, [HYBRID_BETA]*N)
        spikes = [x[(x >= BURN) & (x < BURN+HYBRID_MS)]-BURN for x in spikes]
        counts = binned(spikes, HYBRID_MS)
        trains.append(counts)
        rows, cols = np.triu_indices(N, 1)
        stim_corr.append(np.corrcoef(current[:, BURN*FS:])[rows,cols].mean())
        stim_sd.append(current[:, BURN*FS:].std(axis=1).mean())
        labels = np.digitize(slow, np.quantile(slow[BURN*FS:], [.2,.4,.6,.8]))
        for k, times in enumerate(spikes):
            latest = np.searchsorted(events_i[k], np.arange(len(slow)), side="right") - 1
            eligible = np.ones(len(slow), dtype=bool)
            if len(events_i[k]):
                seen = latest >= 0
                eligible[seen] = np.flatnonzero(seen) - events_i[k][latest[seen]] >= 15*FS
            eligible[:BURN*FS] = False
            sample = np.minimum(((times+BURN)*FS).astype(int), len(slow)-1)
            by_slow += np.bincount(labels[sample[eligible[sample]]], minlength=5)
            exposure += np.bincount(labels[eligible], minlength=5)/(1000*FS)
        t = triggers/FS - BURN
        t = t[(t >= 22) & (t < HYBRID_MS-22)]
        simultaneous = counts.sum(axis=0)
        pair_fraction = simultaneous*(simultaneous-1)/(N*(N-1))
        if len(t):
            event_pairs += pair_fraction[t.astype(int)[:,None] + np.arange(-20,21)].sum(axis=0)
            n_events += len(t)
        if condition == 2 and rep == 0 and len(t):
            start = int(np.clip(t[0]-500, 0, HYBRID_MS-1000))
            sl = slice((BURN+start)*FS, (BURN+start+1000)*FS)
            example = dict(slow=slow[sl], events=t[(t>=start)&(t<start+1000)]-start,
                           raster=counts[:,start:start+1000])
    trains = np.array(trains)
    # Both group positions refer to this one population; retain only its within-type result.
    ccg, rho = summarize(np.stack([trains,trains], axis=1))
    return dict(ccg=ccg[0], rho=rho[0], actual_c=np.mean(stim_corr), input_sd=np.mean(stim_sd),
                rate=trains.sum(axis=-1).mean()*1000/HYBRID_MS, by_slow=by_slow,
                exposure=exposure, event_pairs=event_pairs, n_events=n_events, **example)


def cached_hybrid(args):
    from run import save
    filename, seed, idx, mean = args
    if not filename.exists():
        save(filename, **hybrid_trial(seed, idx, mean), seed=seed, mean=mean, beta=HYBRID_BETA)


def hybrid(workers, cache):
    from run import RESULTS, save, version
    cache = cache / version("probes.py","transfer.py")
    pilot = cache / "pilot.npz"
    if not pilot.exists():
        mean, grid, rates = hybrid_pilot()
        save(pilot, mean=mean, grid=grid, rates=rates)
    with np.load(pilot) as p:
        mean, grid, rates = float(p["mean"]), p["grid"], p["rates"]
    jobs = [(cache/f"hybrid_{s}_{i}.npz", int(s), i, mean)
            for s in HYBRID_SEEDS for i in range(len(CONDITIONS))]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        list(pool.map(cached_hybrid, jobs))
    arrays = {}
    for idx, (path, seed, i, _) in enumerate(jobs):
        with np.load(path) as d:
            for key in ("ccg","rho","actual_c","input_sd","rate","by_slow","exposure","event_pairs","n_events"):
                if key not in arrays:
                    arrays[key] = np.empty((len(CONDITIONS),len(HYBRID_SEEDS))+d[key].shape)
                arrays[key][i,idx//len(CONDITIONS)] = d[key]
            if i == 2 and seed == HYBRID_SEEDS[0]:
                save(RESULTS / "hybrid_example.npz", **{k:d[k] for k in ("slow","events","raster")})
    save(RESULTS / "hybrid.npz", **arrays, mean=mean, grid=grid, pilot_rates=rates,
         beta=HYBRID_BETA, seeds=HYBRID_SEEDS, conditions=CONDITIONS, shares=SHARES)


def run_probes(which, workers, cache):
    if which in ("all","hybrid"):
        hybrid(workers, cache)
    if which in ("all","demo"):
        from demonstration import demo
        demo()
