import numpy as np


def model0(x, a1, a2, tau1, tau2, level):
    model = a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + level
    return model


def model1(x, a1, tau1, tau2, level):
    model = a1 * np.exp(-x / tau1) + (1.0 - a1) * np.exp(-x / tau2) + level
    return model


def model1b(x, a1, tau1, tau2, level=0.0):
    model = a1 * np.exp(-x / tau1) + (1.0 - a1) * np.exp(-x / tau2) + level
    return model
