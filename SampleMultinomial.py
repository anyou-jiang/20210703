import numpy as np

def SampleMultinomial(probabilities):
    dice = np.random.uniform() # returns a uniformly sampled random number from interval (0,1)

    accumulate = 0
    for i in range(len(probabilities)):
        accumulate = accumulate + probabilities[i]
        if accumulate/sum(probabilities) > dice:
            break

    sample = i
    return sample