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

# Set True only when rerunning the original 1800-s FT replication.
RUN_ORIGINAL_FT_REPLICATION = False
if RUN_ORIGINAL_FT_REPLICATION:
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

from pathlib import Path

POP_SIZES = [2, 5, 10, 20]
POP_INPUT_CORRS = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
POP_INPUT_VARIANCE = 2000.0 
POP_SECONDS = 12.0
POP_TRIALS_PER_CORR = 12
POP_SEED = 2027
POP_COUNT_WINDOW_MS = 200
POP_BIN_MS = 1.0
POP_SAVE_DATA = True
POP_DATA_FILE = "ft_population_dataset.npz"


def make_population_currents(c, seed, n_neurons=20, seconds=POP_SECONDS,
                             variance=POP_INPUT_VARIANCE):
    if not (0.0 <= c <= 1.0):
        raise ValueError("Input correlation must be between 0 and 1")
    if n_neurons < 2:
        raise ValueError("At least 2 neurons are needed")

    random = np.random.default_rng(int(seed))
    n_steps = int(round(seconds * 1000.0 / DT_MS))
    decay = np.exp(-DT_MS / TAU_MS)
    scale = np.sqrt(1.0 - decay ** 2)

    common = lfilter([scale], [1.0, -decay], random.standard_normal(n_steps))
    currents = np.empty((n_neurons, n_steps), dtype=np.float64)

    for i in range(n_neurons):
        private = lfilter([scale], [1.0, -decay], random.standard_normal(n_steps))
        currents[i] = np.sqrt(variance) * (
            np.sqrt(c) * common + np.sqrt(1.0 - c) * private
        )
    return currents


def simulate_ft_population(currents):
    n_neurons, n_steps = currents.shape
    warmup_steps = int(round(WARMUP_MS / DT_MS))
    bin_steps = int(round(POP_BIN_MS / DT_MS))
    if warmup_steps >= n_steps or bin_steps < 1:
        raise ValueError("The simulation is too short or bin size is invalid")
    n_bins = (n_steps - warmup_steps) // bin_steps
    if n_bins < POP_COUNT_WINDOW_MS:
        raise ValueError("Not enough time bins for the count window")

    spikes = np.zeros((n_neurons, n_bins), dtype=np.uint8)
    for i in range(n_neurons):
        voltage = lfilter(filter_kernel, [1.0], currents[i])
        raw_spikes = ft_spikes(voltage, THRESHOLD_MV, AHP_MV, ahp_decay)
        trimmed = raw_spikes[warmup_steps:warmup_steps + n_bins * bin_steps]
        spikes[i] = trimmed.reshape(n_bins, bin_steps).sum(axis=1)
    return spikes


def population_corr_matrix(spikes, window=POP_COUNT_WINDOW_MS):
    counts = np.array([moving_counts(s, window) for s in spikes])
    stds = counts.std(axis=1)
    result = np.corrcoef(counts)
    if np.any(stds == 0):
        bad = (stds == 0)
        result[bad, :] = np.nan
        result[:, bad] = np.nan
    return result


def offdiag_mean(matrix):
    idx = np.triu_indices(matrix.shape[0], 1)
    values = matrix[idx]
    if not np.any(np.isfinite(values)):
        return np.nan
    return float(np.nanmean(values))



def calculate_population_ccgs(spikes, input_corr):
    ccg_levels = np.array([0.0, 0.5, 0.9])
    ccg_values = []
    ccg_lags = np.arange(-40, 41)
    n_pairs = spikes.shape[1] // 2

    for c in ccg_levels:
        trial_indices = np.flatnonzero(np.isclose(input_corr, c))
        if len(trial_indices) < 2:
            ccg_values.append(np.full(len(ccg_lags), np.nan))
            continue

        results = []
        for j, idx in enumerate(trial_indices):
            shuffled_idx = trial_indices[(j + 1) % len(trial_indices)]
            for i in range(n_pairs):
                neuron1 = i * 2
                neuron2 = neuron1 + 1
                ccg_lags, current_ccg = get_shift_ccg(
                    spikes[idx, neuron1], spikes[idx, neuron2],
                    spikes[shuffled_idx, neuron2], maxlag=40
                )
                results.append(current_ccg)
        ccg_values.append(np.mean(results, axis=0))
    return ccg_levels, ccg_lags, np.asarray(ccg_values)


def run_ft_population_experiment():
    """Run the extension and return a data dictionary for Part 3."""
    n_max = max(POP_SIZES)
    random = np.random.default_rng(POP_SEED)
    all_spikes = []
    all_corrs = []
    all_seeds = []
    all_actual_input_corr = []
    all_matrices = []
    all_rates = []
    mean_corr_by_size = []

    for c in POP_INPUT_CORRS:
        for trial in range(POP_TRIALS_PER_CORR):
            seed = int(random.integers(0, 2**31 - 1))
            currents = make_population_currents(
                c, seed, n_neurons=n_max,
                seconds=POP_SECONDS, variance=POP_INPUT_VARIANCE
            )
            warmup = int(round(WARMUP_MS / DT_MS))
            actual_c = np.corrcoef(currents[0, warmup:],
                                   currents[1, warmup:])[0, 1]
            spikes = simulate_ft_population(currents)
            matrix = population_corr_matrix(spikes)

            all_spikes.append(spikes)
            all_corrs.append(c)
            all_seeds.append(seed)
            all_actual_input_corr.append(actual_c)
            all_matrices.append(matrix)
            all_rates.append(spikes.mean(axis=1) * (1000.0 / POP_BIN_MS))
            mean_corr_by_size.append([
                offdiag_mean(matrix[:n, :n]) for n in POP_SIZES
            ])

        print(f"Input c = {c:.1f}: {POP_TRIALS_PER_CORR} trials completed")

    spikes_array = np.stack(all_spikes)
    corr_array = np.asarray(all_corrs, dtype=np.float64)
    ccg_levels, ccg_lags, ccg_values = calculate_population_ccgs(
        spikes_array, corr_array
    )

    data = {
        "spikes": spikes_array,
        "input_corr": corr_array,
        "ccg_input_corr": ccg_levels,
        "ccg_lags_ms": ccg_lags,
        "population_ccg_hz2": ccg_values,
        "trial_seed": np.asarray(all_seeds, dtype=np.int64),
        "trial_id": np.arange(len(all_corrs)),
        "measured_input_corr_01": np.asarray(all_actual_input_corr),
        "output_corr_matrix": np.stack(all_matrices),
        "firing_rates_hz": np.stack(all_rates),
        "mean_output_corr_by_size": np.asarray(mean_corr_by_size),
        "pop_sizes": np.asarray(POP_SIZES),
        "neuron_types": np.full(n_max, "FT"),
        "input_variance_pa2": np.float64(POP_INPUT_VARIANCE),
        "input_mean_pa": np.float64(0.0),
        "dt_ms": np.float64(DT_MS),
        "bin_ms": np.float64(POP_BIN_MS),
        "warmup_ms": np.float64(WARMUP_MS),
        "simulation_seconds": np.float64(POP_SECONDS),
        "count_window_ms": np.int64(POP_COUNT_WINDOW_MS),
    }
    return data


def plot_ft_population_results(data):
    """Display population results in Colab; no figure export."""
    corrs = data["input_corr"]
    sizes = data["pop_sizes"]
    mean_corr = data["mean_output_corr_by_size"]
    unique_c = np.unique(corrs)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, n in enumerate(sizes):
        avg = [np.nanmean(mean_corr[corrs == c, j]) for c in unique_c]
        ax[0].plot(unique_c, avg, "o-", label=f"N={n}")
    ax[0].set(xlabel="Input correlation (ground truth c)",
              ylabel="Mean pairwise output correlation",
              title="FT population correlation transfer")
    ax[0].legend()
    ax[0].grid(alpha=0.25)

    target = int(np.argmin(np.abs(corrs - 0.5)))
    matrix = data["output_corr_matrix"][target]
    im = ax[1].imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax[1].set(xlabel="Neuron index", ylabel="Neuron index",
              title=f"Pairwise correlation matrix (c={corrs[target]:.1f})")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for j, n in enumerate(sizes):
        values = mean_corr[:, j]
        spread = [np.nanstd(values[corrs == c]) for c in unique_c]
        ax[0].plot(unique_c, spread, "o-", label=f"N={n}")
    ax[0].set(xlabel="Input correlation (ground truth c)",
              ylabel="SD across trials of mean output correlation",
              title="Population size and estimate variability")
    ax[0].legend()
    ax[0].grid(alpha=0.25)

    ax[1].scatter(corrs, data["measured_input_corr_01"], alpha=0.45)
    ax[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    ax[1].set(xlim=(-0.05, 1), ylim=(-0.1, 1),
              xlabel="Configured input correlation c",
              ylabel="Measured correlation, inputs 1 and 2",
              title="Ground-truth input sanity check")
    ax[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    for c, line in zip(data["ccg_input_corr"], data["population_ccg_hz2"]):
        if np.any(np.isfinite(line)):
            plt.plot(data["ccg_lags_ms"], line, label=f"c={c:.1f}")
    plt.axhline(0, color="gray", linewidth=0.7)
    plt.xlabel("Time lag (ms)")
    plt.ylabel("Excess coincidence (Hz$^2$)")
    plt.title("FT population: shuffle-corrected cross-covariance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nTotal trials:", len(corrs))
    print("Spike array shape:", data["spikes"].shape)
    print("Mean firing rate (Hz):", round(float(data["firing_rates_hz"].mean()), 2))
    print("Input correlation MAE:",
          round(float(np.mean(np.abs(corrs - data["measured_input_corr_01"]))), 4))
    print("Notes: outputs use trial-level Pearson correlations of 200-ms spike counts.")
    print("These are not the shuffle-corrected rho estimates of the original FT plots.")


if __name__ == "__main__":
    pop_data = run_ft_population_experiment()
    plot_ft_population_results(pop_data)
    if POP_SAVE_DATA:
        np.savez_compressed(POP_DATA_FILE, **pop_data)
        print("Dataset saved:", Path(POP_DATA_FILE).resolve())
