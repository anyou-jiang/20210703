import numpy as np
import math

def GaussianMutualInformation(X, Y):
    if np.array_equal(X, Y):
        I = 0
        return I

    Sxx = np.cov(X, rowvar=False)
    Syy = np.cov(Y, rowvar=False)
    S = np.cov(np.concatenate((X, Y), axis=1), rowvar=False)
    I = 0.5 * math.log(np.linalg.det(Sxx) * np.linalg.det(Syy) / np.linalg.det(S))

    return I
