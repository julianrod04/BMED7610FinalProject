"""Regenerate the Part 2 experiments. Run from any directory."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import multiprocessing as mp
from pathlib import Path
import tempfile

import numpy as np

from model import ROOT, build
import transfer as t

RESULTS = ROOT / "results"


def save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
        try:
            np.savez_compressed(f, **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def version(*filenames):
    inputs = [ROOT / "scripts" / f for f in ("model.py", *filenames)] + sorted((ROOT / "mods").glob("*.mod"))
    return sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest()[:12]


def job(path, seed, c, means):
    if not path.exists():
        save(path, **t.gaussian(seed, c, means), seed=seed, c=c, means=means, betas=t.BETAS)
    return path


def gaussian_sweep(cache, workers):
    folder = cache / version("transfer.py")
    pilot = folder / "rate_pilot.npz"
    if not pilot.exists():
        save(pilot, **t.rate_pilot())
    with np.load(pilot) as p:
        means = p['means']
        save(RESULTS / "operating_points.npz", **dict(p))
    tasks = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        for s, seed in enumerate(t.SEEDS):
            for j, c in enumerate(t.LEVELS):
                for i in range(len(t.TARGETS)):
                    path = folder / f"rate{i}_c{j}_seed{seed}.npz"
                    tasks.append(pool.submit(job, path, int(seed), float(c), means[:,i]))
        for done, task in enumerate(as_completed(tasks), 1):
            task.result()
            if done % 48 == 0:
                print(f"{done}/{len(tasks)} paired conditions", flush=True)
    axes = (len(t.TARGETS), len(t.LEVELS), len(t.SEEDS))
    arrays = {}
    for i, j, s in np.ndindex(axes):
        with np.load(folder / f"rate{i}_c{j}_seed{t.SEEDS[s]}.npz") as d:
            for key in ("rho", "ccg", "rate", "actual_c"):
                if key not in arrays:
                    arrays[key] = np.empty(axes + d[key].shape)
                arrays[key][i,j,s] = d[key]
            if i == 1 and j == 3 and s == t.TRAIN:
                save(RESULTS / "pair_example.npz", **{k:d[k] for k in ("input","spikes","offsets")})
    pred = np.empty((len(t.TARGETS), len(t.LEVELS), 3, len(t.LAGS)))
    for i in range(len(t.TARGETS)):
        banks = []
        for group in np.array_split(t.SEEDS[:t.TRAIN], 2):
            sta, counts = np.zeros((2, 200*t.FS+1)), np.zeros(2)
            for seed in group:
                with np.load(folder / f"rate{i}_c0_seed{seed}.npz") as d:
                    sta += d["sta"]
                    counts += d["nsta"]
            banks.append(sta / counts[:, None])
        for j, level in enumerate(t.LEVELS):
            rate = arrays["rate"][i,j,t.TRAIN:].mean(axis=0)
            for k, (a,b) in enumerate([(0,0),(1,1),(0,1)]):
                pred[i,j,k] = np.mean([t.predicted_ccg(banks[x][a], banks[1-x][b],
                                                         rate[a], rate[b], level)
                                        for x in range(2)], axis=0)
    save(RESULTS / "transfer.npz", **arrays, predicted=pred, lags=t.LAGS,
         targets=t.TARGETS, levels=t.LEVELS, seeds=t.SEEDS, train=t.TRAIN, windows=t.WINDOWS,
         pairs=t.PAIRS, means=means, cells=t.N, seconds=t.DURATION / 1000, sd=t.SD)


def legacy(cache):
    rows = []
    for m in np.linspace(345, 375, 12):
        for c in t.LEVELS[1:]:
            path = cache / f"mu{m:g}_sigma2100_c{c:g}_tau5_n_cells100_tstop180100_seed1_dt0.025.npz"
            with np.load(path) as d:
                rows.append([d[k].item() for k in ("mu","c","rate","rho")])
    np.savetxt(RESULTS / "original_integrator.csv", rows, delimiter=",",
               header="mean_pA,input_c,rate_Hz,rho200", comments="", fmt="%.9g")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", type=Path, default=ROOT / ".part2-cache")
    p.add_argument("--legacy-cache", type=Path, help="Optional original integrator .npz folder")
    p.add_argument("--only", choices=["gaussian","hybrid","demo","all"], default="all")
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.legacy_cache:
        legacy(args.legacy_cache)
    build()
    if args.only in ("all","gaussian"):
        gaussian_sweep(args.cache.resolve(), args.workers)
    if args.only in ("all","hybrid","demo"):
        from probes import run_probes
        run_probes(args.only, args.workers, args.cache.resolve())


if __name__ == "__main__":
    main()
