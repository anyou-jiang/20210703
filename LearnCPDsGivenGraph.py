import numpy as np
import copy
from FitGaussianParameters import FitGaussianParameters
from FitLinearGaussianParameters import FitLinearGaussianParameters
from ComputeLogLikelihood import ComputeLogLikelihood

def LearnCPDsGivenGraph(dataset, G, labels):
    #print(dataset.shape)
    #print(G.shape)
    #print(labels.shape)
    N = dataset.shape[0]
    B = dataset.shape[1]
    K = labels.shape[1]

    assert (B == 10)
    assert (K == 2)

    loglikelihood = 0
    P = {}
    P['c'] = np.zeros((K,))
    P['clg'] = []
    P_clg_fields_dict = {'mu_y': np.zeros((K,)),
                         'sigma_y': np.zeros((K,)),
                         'mu_x': np.zeros((K,)),
                         'sigma_x': np.zeros((K,)),
                         'mu_angle': np.zeros((K,)),
                         'sigma_angle': np.zeros((K,)),
                         'theta': np.zeros((K, 12))
                         }
    for b in range(B):
        P['clg'].append(copy.deepcopy(P_clg_fields_dict))

    if len(G.shape) == 2:
        G = np.tile(G.reshape([G.shape[0], G.shape[1], 1]), (1, 1, K))

    count_c1 = np.sum(labels[:, 0])
    count_c2 = np.sum(labels[:, 1])
    P['c'][0] = count_c1 * 1.0 / N
    P['c'][1] = count_c2 * 1.0 / N
    assert(P['c'][0] + P['c'][1] == 1.0)
    
    for i in range(B):
        for k in range(K):
            class_k_idx = np.squeeze(np.asarray(np.where(labels[:, k])))
            if G[i, 0, k] == 0: # case that node i has no parent other than class label
                mu_y, sigma_y = FitGaussianParameters(dataset[class_k_idx, i, 0])
                mu_x, sigma_x = FitGaussianParameters(dataset[class_k_idx, i, 1])
                mu_angle, sigma_angle = FitGaussianParameters(dataset[class_k_idx, i, 2])
                P['clg'][i]['mu_y'][k] = mu_y
                P['clg'][i]['mu_x'][k] = mu_x
                P['clg'][i]['mu_angle'][k] = mu_angle
                P['clg'][i]['sigma_y'][k] = sigma_y
                P['clg'][i]['sigma_x'][k] = sigma_x
                P['clg'][i]['sigma_angle'][k] = sigma_angle
            else: # case that node i has additional parent
                p_i = G[i, 1, k]
                data_parent = dataset[class_k_idx, p_i - 1, :]
                data_parent = np.squeeze(data_parent)
                data_y = dataset[class_k_idx, i, 0]
                Beta_y, sigma_y = FitLinearGaussianParameters(data_y, data_parent)

                data_x = dataset[class_k_idx, i, 1]
                Beta_x, sigma_x = FitLinearGaussianParameters(data_x, data_parent)

                data_angle = dataset[class_k_idx, i, 2]
                Beta_angle, sigma_angle = FitLinearGaussianParameters(data_angle, data_parent)

                P['clg'][i]['sigma_y'][k] = sigma_y
                P['clg'][i]['sigma_x'][k] = sigma_x
                P['clg'][i]['sigma_angle'][k] = sigma_angle

                theta_vec = np.array([Beta_y[-1], Beta_y[0], Beta_y[1], Beta_y[2],
                    Beta_x[-1], Beta_x[0], Beta_x[1], Beta_x[2],
                    Beta_angle[-1], Beta_angle[0], Beta_angle[1], Beta_angle[2]])

                P['clg'][i]['theta'][k, :] = theta_vec[:]

    loglikelihood = loglikelihood + ComputeLogLikelihood(P, G, dataset)
    print('log likelihood: {}'.format(loglikelihood))

    return P, loglikelihood



