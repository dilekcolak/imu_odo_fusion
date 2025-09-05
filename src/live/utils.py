# -*- coding: utf-8 -*-

import numpy as np



def wrap_pi(a):

    return (a + np.pi) % (2*np.pi) - np.pi



class LowPass:

    """Basit düşük geçiren filtre (exponential smoothing)."""

    def __init__(self, alpha, x0=0.0):

        self.a = float(alpha)

        self.y = float(x0)

    def step(self, x):

        self.y = self.a*self.y + (1.0 - self.a)*x

        return self.y

