"""Single-cell controls using the same current waveforms in both ML settings."""
import numpy as np
from model import FS, simulate
from probes import ou
from transfer import BETAS


def pulses(drive, onset_ms, spacing_ms, count=3):
    for j in range(count):
        idx = round(FS*(onset_ms+j*spacing_ms))
        drive[idx:idx+FS] += 980  # pA, 1 ms each


def demo():
    from run import RESULTS, save
    stimulus = np.zeros(260*FS)
    stimulus[25*FS:90*FS] = 650
    pulses(stimulus, 140, 1.4)
    pulses(stimulus, 200, 2.8)
    _, rec = simulate([stimulus]*2, BETAS, record=(0,1))
    spacing = np.array([1.,1.4,1.8,2.2,2.6,3.,3.4,4.,6.])
    probability = np.zeros((2, len(spacing)+1))
    for seed in range(64):
        noise = 25*ou(np.random.default_rng([818,seed]), 120*FS, 5)
        drives = []
        for gap in np.r_[0,spacing]:
            current = noise.copy()
            pulses(current, 70, gap, count=1 if gap == 0 else 3)
            drives.append(current)
        times, _ = simulate(drives*2, np.repeat(BETAS,len(drives)))
        responses = [np.any((s >= 70) & (s < 100)) for s in times]
        probability += np.reshape(responses, probability.shape)/64
    save(RESULTS / "responses.npz", t=np.arange(len(stimulus))/FS, current=rec[0][1],
         voltage=np.array([rec[k][0] for k in range(2)]), spacing=spacing,
         single=probability[:,0], probability=probability[:,1:], trials=64)
