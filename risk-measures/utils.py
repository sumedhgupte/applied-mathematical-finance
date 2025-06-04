import numpy as np
rng = np.random.default_rng()

def unit_projection(x, decimals = 3):
    x = np.around(x, decimals)
    error = 1. - x.sum()
    x[x.argmax()]+= error
    return x

def draw_noisy_samples(X, m):
    var = np.diag(np.cov(X, rowvar=False))
    return X[rng.choice(X.shape[0], m)] + rng.normal(0, var/2, size=(m, var.size))

def draw_samples(X, m):
    return X[rng.choice(X.shape[0], m)] 