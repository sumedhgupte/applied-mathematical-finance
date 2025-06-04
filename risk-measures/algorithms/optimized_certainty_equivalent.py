from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import pyproximal
from joblib import Parallel, delayed
from algorithms.base import Risk, Objective
from utils import draw_samples, unit_projection, draw_noisy_samples
rng = np.random.default_rng()

@dataclass
class OCE:
    utility_fn : np.ufunc 
    utility_derivative : np.ufunc 

    def g(self, X, t):
        return self.utility_derivative(-X-t).mean() - 1
        
    def G(self, X, t):
        return t + self.utility_fn(-X-t).mean() 

    def OCE_SB(self, Z, delta, epsilon):
        sign = np.sign(self.g(Z, 0))
        low, high = min(0, sign), max(0, sign)
        while self.g(Z, high) > 0:
            high = high * 2
        while self.g(Z, low) < 0:
            low = low * 2
        t = (high + low) / 2
        while ((high - low) > (2 * delta)) or np.abs(self.g(Z, t)) > epsilon:
            if self.g(Z, t) > 0:
                low = t
            else:
                high = t
            t = (high + low) / 2
        return t

    def OCE_SAA(self, Z, delta, epsilon): 
        root_estimate = self.OCE_SB(Z, delta, epsilon)
        return root_estimate, self.G(Z, root_estimate)
        
@dataclass
class OCE_Risk(Risk):
    utility_fn : np.ufunc
    utility_derivative : np.ufunc
    delta : np.float32
    epsilon : np.float32
    objective : Objective
    projection : any

    def fn(self, x):
        return self.utility_fn(x)

    def g(self, t, X):
        return self.utility_derivative(-X-t).mean() - 1

    def OCE_SB(self, Z, delta, epsilon):
        sign, counter = np.sign(self.g(0, Z)), 0
        low, high = min(0, sign), max(0, sign)
        while self.g(high, Z) > 0:
            high = high * 2
        while self.g(low, Z) < 0:
            low = low * 2
        t = (high + low) / 2
        while (high - low) > (2 * delta) or np.abs(self.g(t, Z))> epsilon:
            if counter > 40: 
                print('overtime')
                break
            counter += 1
            if self.g(t, Z) > 0:
                low = t
            else:
                high = t
            t = (high + low) / 2
        return t

    def OCE_SAA(self, Z, delta, epsilon): 
        root_estimate = self.OCE_SB(Z, delta, epsilon)
        return root_estimate, self.G(Z, root_estimate)

    def OCE_SG(self, X, epochs):
        '''
        X : 2D nparray of size: number of samples (R) * data dimension (d)
        '''
        R, d = X.shape[0], X.shape[1]
        theta = np.ones(d)/d
        for k in tqdm(range(1, epochs+1)):
            m = k
            Z_hat, Z = draw_noisy_samples(X, m), draw_noisy_samples(X, m)
            sr_k = self.OCE_SB(self.objective.F(theta, Z_hat), self.delta/np.sqrt(k), self.epsilon)
            u_prime_k = self.utility_derivative(-self.objective.F(Z, theta) - sr_k)
            grad = -(self.objective.grad_F(theta, Z).T * u_prime_k).mean(axis=-1)
            alpha = 1. / np.sqrt(k)
            projection = pyproximal.projection.SimplexProj(d, radius=1, maxiter = 2000, xtol = 1e-6)
            theta = unit_projection(projection(theta  - alpha * grad), np.ceil(np.log10(d)).astype(int))
            #theta  = self.projection(theta  - alpha * grad)
        return theta

    def fit(self, X, epochs, simulations = 5):
        return np.mean(Parallel(n_jobs=-2)(delayed(self.OCE_SG)(X, epochs) for i in range(simulations)), axis = 0)