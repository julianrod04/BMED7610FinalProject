"""Single-compartment Morris–Lecar cells with replayable injected currents."""
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("PART2_BUILD", ROOT / "build")).resolve()
FS = 5  # stimulus samples/ms
DT = 0.025  # integration step in ms
TAU = 5.0


def build():
    libs = list(BUILD.glob("*/libnrnmech.*"))
    sources = list((ROOT / "mods").glob("*.mod"))
    if libs and min(f.stat().st_mtime for f in libs) >= max(f.stat().st_mtime for f in sources):
        return
    BUILD.mkdir(parents=True, exist_ok=True)
    exe = Path(sys.executable).parent / "nrnivmodl"
    subprocess.run([str(exe) if exe.exists() else shutil.which("nrnivmodl") or "nrnivmodl",
                    str(ROOT / "mods")], cwd=BUILD, check=True)


@lru_cache(maxsize=1)
def engine():
    import neuron
    from neuron import h
    if not hasattr(h, "CIClamp"):
        neuron.load_mechanisms(str(BUILD))
    h.usetable_ml = 0  # the original rate lookup table does not depend on beta_w
    return h


def simulate(drives, betas, *, means=None, sds=None, record=()):
    """With means/sds, drive CIClamp using white innovations. Otherwise replay pA directly."""
    h = engine()
    n = len(drives[0])
    duration = n / FS
    keep, counts, traces = [], [], {}
    for k, (drive, beta) in enumerate(zip(drives, betas, strict=True)):
        assert len(drive) == n
        cell = h.Section(name=f"soma{k}")
        cell.diam, cell.L, cell.cm = 25, 40 / np.pi, 2
        cell.insert("ml")
        cell.ek, cell.ena = -100, 50
        cell(0.5).ml.betaw = beta
        if means is None:
            stim = h.IClamp(cell(0.5))
            stim.delay, stim.dur = 0, duration + 1
            values = h.Vector(drive * 1e-3)
        else:
            stim = h.CIClamp(cell(0.5))
            stim.tcorr, stim.mu = TAU, means[k] * 1e-3
            stim.gain = sds[k] * 1e-3 * np.sqrt(2 * FS / TAU)
            values = h.Vector(drive)
        values.play(stim._ref_amp, 1 / FS)
        counter = h.APCount(cell(0.5))
        counter.thresh = -10
        times = h.Vector()
        counter.record(times)
        counts.append(times)
        if k in record:
            traces[k] = (h.Vector().record(cell(0.5)._ref_v, 1 / FS),
                         h.Vector().record(stim._ref_i, 1 / FS))
        keep.append((cell, stim, values, counter))
    h.dt = DT
    pc = h.ParallelContext()
    pc.set_maxstep(10)
    h.finitialize(-65)
    pc.psolve(duration)
    spikes = [np.array(v) for v in counts]
    traces = {k: (np.array(v)[:n], np.array(i)[:n] * 1000) for k, (v, i) in traces.items()}
    return spikes, traces
