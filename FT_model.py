import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, correlate
from numba import njit

##basic Parameters
DT_MS = 0.2
TAU_MS = 5.0
FILTER_MS = 15.0
THRESHOLD_MV = 1.0
AHP_MV = -0.5
AHP_TAU_MS = 30.0
INPUT_CORR = 0.5
INPUT_MEAN_PA = 0.0
INPUT_VARIANCES = np.linspace(100.0, 7000.0, 30)
COUNT_WINDOW_MS = 200
SIM_SECONDS = 1800
REPEATS = 4
WARMUP_MS = 200
CCG_EXAMPLES = (0, 14, 29)
SEED = 2026
rng = np.random.default_rng(SEED)

#buit biphasic Blackman derivative filter
points = int(round(FILTER_MS / DT_MS))
filter_kernel = np.gradient(
    np.blackman(points), DT_MS
)
filter_kernel -= filter_kernel.mean()
t = np.arange(points)
R = np.exp(
    -np.abs(t[:, None] - t[None, :])
    * DT_MS / TAU_MS
)
filter_kernel *= np.sqrt(
    0.1 / (filter_kernel @ R @ filter_kernel)
)

ou_decay = np.exp(-DT_MS / TAU_MS)
ahp_decay = np.exp(-DT_MS / AHP_TAU_MS)
steps = int(round(SIM_SECONDS * 1000 / DT_MS))
burn = int(round(WARMUP_MS / DT_MS))
per_ms = int(round(1.0 / DT_MS))
rng = np.random.default_rng(SEED)

def filtered_ou():
    noise = rng.standard_normal(steps)
    ou = lfilter(
        [np.sqrt(1.0 - ou_decay**2)],
        [1.0, -ou_decay],
        noise
    )
    return lfilter(filter_kernel, [1.0], ou)


@njit
def ft_spikes(voltage, threshold, ahp_step, decay):
    spikes = np.zeros(len(voltage), dtype=np.uint8)
    ahp = 0.0
    previous = -1e9

    for i in range(len(voltage)):
        ahp *= decay
        crossed = (previous < threshold and voltage[i] >= threshold)

        if crossed and voltage[i] + ahp >= threshold:
            spikes[i] = 1
            ahp += ahp_step

        previous = voltage[i]

    return spikes


def bin_spikes(spikes):
    s = spikes[burn:]
    s = s[:(len(s) // per_ms) * per_ms]
    return s.reshape(-1, per_ms).sum(axis=1).astype(np.uint8)


def moving_counts(binned, window=COUNT_WINDOW_MS):
    cumulative = np.r_[0.0, np.cumsum(binned, dtype=np.float64)]
    return cumulative[window:] - cumulative[:-window]


def time_cov(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return np.mean(x * y)


def get_shift_ccg(a, b, b_shift, maxlag=40):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    b_shift = b_shift.astype(np.float64)

    lags = np.arange(-maxlag, maxlag + 1)
    center = len(a) - 1

    same = correlate(a, b, mode="full", method="fft")
    shifted = correlate(a, b_shift, mode="full", method="fft")

    normalizer = len(a) - np.abs(lags)
    ccg = (same[center + lags] - shifted[center + lags]) / normalizer
    return lags, ccg / (0.001**2)  # counts/bin² -> Hz²

rates = np.zeros((REPEATS, len(INPUT_VARIANCES)))
trains1 = [[] for _ in INPUT_VARIANCES]
trains2 = [[] for _ in INPUT_VARIANCES]
example_trace = None

for repeat in range(REPEATS):
    common = filtered_ou()
    private1 = filtered_ou()
    private2 = filtered_ou()

    base1 = np.sqrt(INPUT_CORR) * common + np.sqrt(1 - INPUT_CORR) * private1
    base2 = np.sqrt(INPUT_CORR) * common + np.sqrt(1 - INPUT_CORR) * private2

    for j, variance in enumerate(INPUT_VARIANCES):
        voltage1 = np.sqrt(variance) * base1
        voltage2 = np.sqrt(variance) * base2
        spikes1 = ft_spikes(voltage1, THRESHOLD_MV, AHP_MV, ahp_decay)
        spikes2 = ft_spikes(voltage2, THRESHOLD_MV, AHP_MV, ahp_decay)
        b1 = bin_spikes(spikes1)
        b2 = bin_spikes(spikes2)
        trains1[j].append(b1)
        trains2[j].append(b2)
        rates[repeat, j] = 0.5 * (b1.mean() + b2.mean()) * 1000.0
        if example_trace is None and j == 14:
            example_trace = (voltage1.copy(), spikes1.copy())

    print(f"Completed repeat {repeat + 1}/{REPEATS}")

mean_rate = rates.mean(axis=0)
mean_rho = np.zeros(len(INPUT_VARIANCES))
ccgs = {j: [] for j in CCG_EXAMPLES}
for j in range(len(INPUT_VARIANCES)):
    x = [moving_counts(b) for b in trains1[j]]
    y = [moving_counts(b) for b in trains2[j]]
    same12 = np.mean([time_cov(x[k], y[k]) for k in range(REPEATS)])
    shift12 = np.mean([
        time_cov(x[k], y[(k + 1) % REPEATS])
        for k in range(REPEATS)
    ])

    auto1 = np.mean([time_cov(x[k], x[k]) for k in range(REPEATS)])
    auto1_shift = np.mean([
        time_cov(x[k], x[(k + 1) % REPEATS])
        for k in range(REPEATS)
    ])
    auto2 = np.mean([time_cov(y[k], y[k]) for k in range(REPEATS)])
    auto2_shift = np.mean([
        time_cov(y[k], y[(k + 1) % REPEATS])
        for k in range(REPEATS)
    ])

    cross_cov = same12 - shift12
    var1 = auto1 - auto1_shift
    var2 = auto2 - auto2_shift

    if var1 > 0 and var2 > 0:
        mean_rho[j] = cross_cov / np.sqrt(var1 * var2)
    else:
        mean_rho[j] = np.nan

    if j in CCG_EXAMPLES:
        for k in range(REPEATS):
            lags, ccg = get_shift_ccg(
                trains1[j][k], trains2[j][k],
                trains2[j][(k + 1) % REPEATS]
            )
            ccgs[j].append(ccg)

rhos = mean_rho.copy()

valid = np.isfinite(mean_rho)
trend = np.corrcoef(
    mean_rate[valid], (mean_rho / INPUT_CORR)[valid]
)[0, 1]

print("Firing-rate range (Hz):", np.round([mean_rate.min(), mean_rate.max()], 3))
print("rho/c range:", np.round(np.nanmin(mean_rho / INPUT_CORR), 3),
      np.round(np.nanmax(mean_rho / INPUT_CORR), 3))
print("Pearson r across input variances:", round(float(trend), 3))

mu_values = np.array(
    [-100, -50, 0, 50, 100],
    dtype=float
)
sigma_for_mu = np.sqrt(2000.0)
mu_rate = []

for mu in mu_values:
    dc_filtered = lfilter(
        filter_kernel,
        [1.0],
        np.full(steps, mu)
    )

    sp = ft_spikes(
        sigma_for_mu * base1 + dc_filtered,
        THRESHOLD_MV,
        AHP_MV,
        ahp_decay
    )

    mu_rate.append(
        bin_spikes(sp).mean() * 1000.0
    )

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# plot1: FT filter
filter_time = np.arange(
    len(filter_kernel)
) * DT_MS

ax[0, 0].plot(
    filter_time,
    filter_kernel
)

ax[0, 0].axhline(
    0, color="gray", linewidth=1
)

ax[0, 0].set(
    xlabel="Time (ms)",
    ylabel="Filter weight (mV/pA)",
    title="A: Biphasic FT filter"
)
# plot2: Rate versus mean
ax[0, 1].plot(
    mu_values,
    mu_rate,
    "o-"
)

ax[0, 1].set(
    xlabel="Input mean (pA)",
    ylabel="Firing rate (Hz)",
    title="B: Rate vs. input mean"
)


# plot3: Rate versus variance
ax[1, 0].plot(
    INPUT_VARIANCES,
    mean_rate,
    "o-",
    markersize=3
)

ax[1, 0].set(
    xlabel="Input variance (pA²)",
    ylabel="Firing rate (Hz)",
    title="B: Rate vs. input variance"
)

# plot4: Correlation susceptibility
ax[1, 1].plot(
    mean_rate,
    mean_rho / INPUT_CORR,
    "o-",
    markersize=4
)

ax[1, 1].set(
    xlabel="Firing rate (Hz)",
    ylabel="Correlation susceptibility ρ/c",
    title="C: Output correlation vs. rate"
)

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4.5))
for j in CCG_EXAMPLES:
    ccg_mean = np.mean(ccgs[j], axis=0)
    plt.plot(
        lags,
        ccg_mean,
        label=(
            f"variance = {INPUT_VARIANCES[j]:.0f} pA², "
            f"rate = {mean_rate[j]:.1f} Hz"
        )
    )
plt.axhline(
    0,
    color="black",
    linewidth=0.8
)
plt.xlabel("Time lag (ms)")
plt.ylabel("Excess coincidence (Hz²)")
plt.title(
    "C: FT spike-train cross-covariance"
)

plt.legend()
plt.tight_layout()
plt.show()

v, sp = example_trace
start = burn + int(0.5 * 1000 / DT_MS)
width = int(150 / DT_MS)
local_t = np.arange(width) * DT_MS
plt.figure(figsize=(11, 3.5))
plt.plot(
    local_t,
    v[start:start + width],
    linewidth=1
)
plt.axhline(
    THRESHOLD_MV,
    color="black",
    linestyle="--",
    label="Threshold"
)
sp_times = local_t[
    sp[start:start + width] > 0
]
plt.vlines(
    sp_times,
    THRESHOLD_MV,
    THRESHOLD_MV + 2,
    color="red",
    linewidth=1.5,
    label="Spikes"
)
plt.xlabel("Time (ms)")
plt.ylabel("Filtered voltage (mV)")
plt.title(
    "FT model: filtered input and spike events"
)
plt.legend()
plt.tight_layout()
plt.show()