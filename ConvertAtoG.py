import numpy as np

def ConvertAtoG(A):
    #print(A.shape)
    G = np.zeros((10, 2)).astype('int')
    A = A + A.transpose()

    G[0, :] = np.array([0, 0])
    visited = np.zeros((10,))
    visited[0] = 1

    cnt = 0
    while np.sum(visited) < 10:
        cnt = cnt + 1
        for i in range(1, 10):
            for j in range(10):
                if (A[i, j] == 1) and (visited[j] == 1):
                    visited[i] = 1
                    G[i, 0] = 1
                    G[i, 1] = j + 1 # index from 1 to align with matlab
                    break

    return G

