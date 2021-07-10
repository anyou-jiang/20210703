import numpy as np

def MaxVertex(Q, keys):
    val = np.amax(keys[Q])
    index = np.argmax(keys[Q])
    vertex = Q[index]
    return index, vertex

def AdjFromPreds(predecessors, n):
    adj = np.zeros((n,n))
    for i in range(n):
        if predecessors[i] != -1:
            adj[predecessors[i], i] = 1

    return adj

def MaxSpanningTree(weights):
    #print(weights.shape)
    n = weights.shape[0]
    assert(weights.shape[0] == weights.shape[1])

    keys = float('-inf') * np.ones((n,))
    predecessors = -1 * np.ones((n,)).astype('int')
    root = 0
    keys[root] = 0

    Q = [x for x in range(n)] # Queue of vertices to put in the tree

    while not (len(Q) == 0):
        index, vertex = MaxVertex(Q, keys)
        remover = np.concatenate((np.arange(0, index), np.arange(index + 1, len(Q))))
        Q = [Q[t] for t in remover.tolist()]
        for v in range(n):
            if (v in Q) and (weights[vertex, v] > keys[v]):
                predecessors[v] = vertex
                keys[v] = weights[vertex, v]

    adj = AdjFromPreds(predecessors, n)

    return adj


