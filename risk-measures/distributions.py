from dataclasses import dataclass
import numpy as np
import scipy
rng = np.random.default_rng()

@dataclass
class Distribution:
    def quantile(self, alpha): pass
    def superquantile(self, alpha): pass
    def sample(self, size): pass

from scipy.special import ndtri, erfinv, erf, gamma, gammainc, expi
from scipy.stats import norm, laplace, logistic, t, genextreme, pareto

@dataclass
class Gaussian:
    mu : np.float32
    std : np.float32 

    def quantile(self, alpha):
        return norm.ppf(alpha, loc=self.mu, scale=self.std)

    def superquantile(self, alpha):
        return -self.mu + (self.std*norm.pdf(norm.ppf(alpha)))/(1-alpha)

    def sample(self, size):
        return norm.rvs(loc=self.mu, scale=self.std, size=size)


@dataclass
class Student_t:
    df : np.int8
    mu : np.float32
    std : np.float32

    def quantile(self, alpha):
        return t.ppf(alpha, df = self.df, loc = self.mu, scale = self.std)

    def superquantile(self, alpha):
        return -self.mu + self.std * (t.pdf(t.ppf(alpha, df = self.df), df = self.df) * (self.df +t.ppf(alpha, df = self.df)**2)) / ((self.df - 1)*(1-alpha))

    def sample(self, size):
        return t.rvs(df=self.df, loc = self.mu, scale = self.std, size=size)

@dataclass
class Laplace:
    mu: np.float32
    beta: np.float32

    def quantile(self, alpha):
        return laplace.ppf(alpha, loc=self.mu, scale=self.beta) 

    def superquantile(self, alpha):
        return np.where(alpha<0.5, -self.mu + (self.beta*alpha*(1-np.log(2*alpha)))/(1-alpha), -self.mu + self.beta*(1-np.log(2*(1-alpha))))
        
    def sample(self, size):
        return laplace.rvs(loc = self.mu, scale = self.beta, size = size)
        
@dataclass
class Pareto:
    beta: np.float32

    def quantile(self, alpha):
        return pareto.ppf(alpha, b=self.beta)

    def superquantile(self, alpha): 
        if self.beta != 0.:
            return -self.mu + 8
        
    def sample(self, size):
        return pareto.rvs(b=self.beta, size = size)

@dataclass
class Logistic:
    mu : np.float32
    beta : np.float32

    def quantile(self, alpha):
        return logistic.ppf(alpha, loc=self.mu,scale=self.beta) 
        
    def superquantile(self, alpha):
        return -self.mu + self.beta * (-alpha*np.log(alpha)-(1-alpha)*np.log(1-alpha))/(1-alpha)

    def sample(self, size):
        return logistic.rvs(loc=self.mu, scale=self.beta, size=size)

@dataclass
class GEV:
    mu : np.float32
    std: np.float32
    shape: np.float32

    def quantile(self, alpha):
        return genextreme.ppf(alpha, loc=self.mu, scale = self.std, c=self.shape)

    def superquantile(self, alpha):
        if self.shape !=0:
            return -self.mu + ((self.std * gamma(1-self.shape)) / (self.shape * (alpha))) * (gammainc(1-self.shape, np.log(1/alpha)) - (1-alpha))
        return -self.mu + (self.std / (1-alpha)) * (np.euler_gamma + alpha * np.log(-np.log(alpha)) - expi(np.log(alpha)))

    def sample(self, size):
        return genextreme.rvs(loc=self.mu, scale = self.std, c = self.shape, size = size)