import numpy as np
import math

def FitGaussianParameters(X):
    '''
    :param X:
    :return:
    '''

    mu = 0
    sigma = 1

    nSamples = X.shape[0]
    sum_X = np.sum(X, axis=0)
    mu = sum_X / nSamples

    sum_X_squared = np.dot(X, X)
    mu_squared = sum_X_squared / nSamples

    var = mu_squared - mu ** 2
    sigma = math.sqrt(var)

    return mu, sigma




