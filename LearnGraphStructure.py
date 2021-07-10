import numpy as np
from GaussianMutualInformation import GaussianMutualInformation
from MaxSpanningTree import MaxSpanningTree

def LearnGraphStructure(dataset):
    #print(dataset.shape)

    N = dataset.shape[0]
    B = dataset.shape[1]
    K = dataset.shape[2]

    assert(B == 10)
    assert(K == 3)

    W = np.zeros((10, 10))

    for i in range(B):
        Oi = dataset[:, i, :]
        for j in range(i, B): # TODO: range(i+1, B) vs. range(i, B)
            Oj = dataset[:, j, :]
            MI = GaussianMutualInformation(Oi, Oj)
            W[i, j] = MI

    add_tran_W = W + W.transpose()
    off_diagonal_W = np.multiply(np.eye(B), W) # TODO: is this step necessary?
    assert(np.array_equal(off_diagonal_W, np.zeros((10, 10))))

    W = add_tran_W - off_diagonal_W
    A = MaxSpanningTree(W)

    return A, W






