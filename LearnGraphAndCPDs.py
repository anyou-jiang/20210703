import numpy as np

from LearnGraphStructure import LearnGraphStructure
from ConvertAtoG import ConvertAtoG
from LearnCPDsGivenGraph import LearnCPDsGivenGraph

def LearnGraphAndCPDs(dataset, labels):
    N = dataset.shape[0]
    K = labels.shape[1]
    B = dataset.shape[1]
    assert(B == 10)

    G = np.zeros((10, 2, K)).astype("int")
    # beside the first row (the first body part, torso) which has not parent, the others all have parents
    for k in range(K):
        G[1:10, :, k] = 1 # G[i, 0] = 1 to indicate that body part i has, besides the class variable, another parent, G[i, 1],TODO: G[1:10, 0, k] = 1?

    for k in range(K):
        dataset_classK = dataset[(labels[:, k] == 1), :, :]
        A_k, _ = LearnGraphStructure(dataset_classK)
        G[:, :, k] = ConvertAtoG(A_k)

    P, likelihood = LearnCPDsGivenGraph(dataset, G, labels)
    return P, G, likelihood

