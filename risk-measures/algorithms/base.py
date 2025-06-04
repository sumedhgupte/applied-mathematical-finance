from enum import Enum
from dataclasses import dataclass
import numpy as np
import plotly.express as px

class RISK_TYPE(Enum):
    UBSR = "loss function"
    OCE = "utility function"
    
@dataclass
class Risk:
    name : str
    fn_type : RISK_TYPE 

    def fn(self, x):pass

@dataclass 
class Objective:
    def F(theta, z): pass
    def grad_F(theta, z) : pass

@dataclass
class Portfolio_Optimization(Objective):
    
    def F(theta, z):
        return np.dot(theta, z.T)
    
    def grad_F(theta, z):
        return z

@dataclass
class Trainer:
    risk : Risk
    epochs : int 
    x_low : np.float32 = -5
    x_high : np.float32 = 5

    def train(self, df):
        self.weights_ = self.risk.fit(df.to_numpy(), self.epochs, simulations= 1)

    def get_figure(self):
        x = np.linspace(self.x_low, self.x_high, 100)
        fig = px.line(x=x, y=self.risk.fn(x), 
                      title=r'Plotting '+str(self.risk.fn_type)+'function from x='+str(self.x_low)+' to x='+str(self.x_high),
                     width = 400, height = 400)
        fig.update_layout(
            xaxis_title=r'$x$',
            yaxis_title=str(self.risk.fn_type)+'(x)')
        return fig