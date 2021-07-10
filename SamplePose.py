import numpy as np
from SampleMultinomial import SampleMultinomial
from SampleGaussian import SampleGaussian

def SamplePose(P, G, label):
    sample = np.zeros((10, 3))

    if label == -1:
        k = SampleMultinomial(P['c'])
    else:
        k = label

    visited = np.zeros((10,))

    while sum(visited) < 10:
        for i in range(10):
            par = G[i, 1] #  parent body part, if exists
            if G[i, 0] == 0:
                muy = P['clg'][i]['mu_y'][k]
                mux = P['clg'][i]['mu_x'][k]
                muangle = P['clg'][i]['mu_angle'][k]
                sample[i, 0] = SampleGaussian(muy, P['clg'][i]['sigma_y'][k])
                sample[i, 1] = SampleGaussian(mux, P['clg'][i]['sigma_x'][k])
                sample[i, 2] = SampleGaussian(muangle, P['clg'][i]['sigma_angle'][k])
                visited[i] = 1
            elif G[i, 0] == 1:
                if visited[par-1] == 1:
                    muy = P['clg'][i]['theta'][k][0] \
                          + P['clg'][i]['theta'][k][1] * sample[par-1, 0] \
                          + P['clg'][i]['theta'][k][2] * sample[par-1, 1] \
                          + P['clg'][i]['theta'][k][3] * sample[par-1, 2]
                    mux = P['clg'][i]['theta'][k][4] \
                          + P['clg'][i]['theta'][k][5] * sample[par-1, 0] \
                          + P['clg'][i]['theta'][k][6] * sample[par-1, 1] \
                          + P['clg'][i]['theta'][k][7] * sample[par-1, 2]
                    muangle = P['clg'][i]['theta'][k][8] \
                          + P['clg'][i]['theta'][k][9] * sample[par-1, 0] \
                          + P['clg'][i]['theta'][k][10] * sample[par-1, 1] \
                          + P['clg'][i]['theta'][k][11] * sample[par-1, 2]
                    sample[i, 0] = SampleGaussian(muy, P['clg'][i]['sigma_y'][k])
                    sample[i, 1] = SampleGaussian(mux, P['clg'][i]['sigma_x'][k])
                    sample[i, 2] = SampleGaussian(muangle, P['clg'][i]['sigma_angle'][k])
                    visited[i] = 1
            elif G[i, 0] == 2:
                raise Exception('Paramerization by (8,9,10) is not supported yet in exercise 2')

    return sample
