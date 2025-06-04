from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import pyproximal
from joblib import Parallel, delayed
from algorithms.base import Risk, Objective
from utils import draw_samples, unit_projection, draw_noisy_samples
rng = np.random.default_rng()

@dataclass
class SR:
    loss_fn : np.ufunc
    threshold : np.float32

    def g(self, t, X):
        return self.loss_fn(-X-t).mean() - self.threshold

    def UBSR_SB(self, Z, delta):
        sign = np.sign(self.g(0, Z))
        low, high = min(0, sign), max(0, sign)
        while self.g(high, Z) > 0:
            high = high * 2
        while self.g(low, Z) < 0:
            low = low * 2
        t = (high + low) / 2
        while (high - low) > (2 * delta):
            if self.g(t, Z) > 0:
                low = t
            else:
                high = t
            t = (high + low) / 2
        return t

@dataclass
class UBSR(Risk):
    loss_fn : np.ufunc
    loss_derivative : np.ufunc
    threshold : np.float32
    delta : np.float32
    objective : Objective 
    projection : any

    def fn(self, x):
        return self.loss_fn(x)

    def g(self, t, X):
        return self.loss_fn(-X-t).mean() - self.threshold

    def UBSR_SB(self, Z):
        delta = self.delta / np.sqrt(Z.size)
        sign = np.sign(self.g(0, Z))
        low, high = min(0, sign), max(0, sign)
        while self.g(high, Z) > 0:
            high = high * 2
        while self.g(low, Z) < 0:
            low = low * 2
        t = (high + low) / 2
        while (high - low) > (2 * delta):
            if self.g(t, Z) > 0:
                low = t
            else:
                high = t
            t = (high + low) / 2
        return t

    def UBSR_SG(self, X, epochs):
        '''
        X : 2D nparray of size: number of samples (R) * data dimension (d)
        '''
        R, d = X.shape[0], X.shape[1]
        theta = np.ones(d)/d
        for k in tqdm(range(1, epochs+1)):
            m = k
            Z_hat, Z = draw_samples(X, m), draw_samples(X, m)
            sr_k = self.UBSR_SB(self.objective.F(theta, Z_hat))
            l_prime_k = self.loss_derivative(-self.objective.F(theta, Z) - sr_k)
            grad = -(self.objective.grad_F(theta, Z).T * l_prime_k).mean(axis=-1) / l_prime_k.mean()
            alpha = 1. / np.sqrt(k)
            #theta  = self.projection(theta  - alpha * grad)
            projection = pyproximal.projection.SimplexProj(d, radius=1, maxiter = 2000, xtol = 1e-6)
            theta = unit_projection(projection(theta  - alpha * grad), np.ceil(np.log10(d)).astype(int))
        return theta

    def fit(self, X, epochs, simulations = 5):
        return np.mean(Parallel(n_jobs=-2)(delayed(self.UBSR_SG)(X, epochs) for i in range(simulations)), axis = 0)