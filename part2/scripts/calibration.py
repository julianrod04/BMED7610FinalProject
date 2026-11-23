"""Out-of-sample inversion of same-type correlation-transfer curves."""
import numpy as np


def slopes(train, levels):
    # One through-origin least-squares line per rate stratum, pairing and window.
    mean_rho = train.mean(axis=2)
    return np.einsum("c,rcpw->rpw", levels, mean_rho) / (levels @ levels)


def errors(test, fitted, levels, source, target):
    estimates = test[:,:,:,target,:] / fitted[:,None,None,source,:]
    return estimates - levels[None,:,None,None]


def evaluate(rho, levels, boundary, repetitions=1000):
    train, test = rho[:,:,:boundary], rho[:,:,boundary:]
    fitted = slopes(train, levels)
    comparisons = ((0,0),(0,2),(1,1),(1,2))
    def summarize(training, testing):
        fit = slopes(training, levels)
        err = np.array([errors(testing, fit, levels, a, b) for a,b in comparisons])
        return np.stack([err.mean(axis=(1,2,3)), np.abs(err).mean(axis=(1,2,3))],axis=-1)
    point = summarize(train, test)
    rng = np.random.default_rng(81)
    bootstrap = []
    for _ in range(repetitions):
        a = rng.integers(train.shape[2], size=train.shape[2])
        b = rng.integers(test.shape[2], size=test.shape[2])
        bootstrap.append(summarize(train[:,:,a], test[:,:,b]))
    return fitted, point, np.quantile(bootstrap, [.025,.975], axis=0)
