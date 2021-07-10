import numpy as np

def SampleGaussian(mu, sigma):
    sample = mu + sigma * np.random.standard_normal()
    return sample