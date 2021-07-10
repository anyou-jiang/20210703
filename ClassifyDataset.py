import numpy as np
import math

from lognormpdf import lognormpdf

def ClassifyDataset(dataset, labels, P, G):
    #print(dataset.shape)
    #print(G.shape)
    N = dataset.shape[0]
    K = labels.shape[1]
    B = dataset.shape[1]

    labels_logical = (labels == 1)
    unique_rows = np.unique(labels_logical, axis=0)
    mate_row = np.sum(unique_rows.astype('int'), axis=0)

    if len(G.shape) == 2:
        G = np.tile(G.reshape([G.shape[0], G.shape[1], 1]), (1, 1, K))

    # class_idxs = np.zeros((N,))
    # for i in range(N):
    #     if labels[i, 0] == 1:
    #         class_idxs[i] = 1
    #     else:
    #         class_idxs[i] = 2

    class_prob = np.zeros((N, K))
    class_pred = np.zeros((N, K))

    for n in range(N):
        O = dataset[n, :, :]
        for k in range(K):
            P_ck = P['c'][k]
            sum_lognorm_probs = 0
            for i in range(B):
                Oi = O[i, :]
                y_sigma = P['clg'][i]['sigma_y'][k]
                x_sigma = P['clg'][i]['sigma_x'][k]
                angle_sigma = P['clg'][i]['sigma_angle'][k]

                y_i = Oi[0]
                x_i = Oi[1]
                angle_i = Oi[2]

                if G[i, 0, k] == 0: # no parents other than class
                    y_mu = P['clg'][i]['mu_y'][k]
                    x_mu = P['clg'][i]['mu_x'][k]
                    angle_mu = P['clg'][i]['mu_angle'][k]
                else:
                    p_i = G[i, 1, k]
                    y_p_i = O[p_i-1, 0]
                    x_p_i = O[p_i-1, 1]
                    angle_p_i = O[p_i-1, 2]
                    p_i_vals = np.array([1, y_p_i, x_p_i, angle_p_i])

                    y_mu = np.dot(p_i_vals, P['clg'][i]['theta'][k, 0:4])
                    x_mu = np.dot(p_i_vals, P['clg'][i]['theta'][k, 4:8])
                    angle_mu = np.dot(p_i_vals, P['clg'][i]['theta'][k, 8:12])

                log_p_Oi_y = lognormpdf(y_i, y_mu, y_sigma)
                log_p_Oi_x = lognormpdf(x_i, x_mu, x_sigma)
                log_p_Oi_angle = lognormpdf(angle_i, angle_mu, angle_sigma)
                log_p_Oi = log_p_Oi_y + log_p_Oi_x + log_p_Oi_angle
                sum_lognorm_probs = sum_lognorm_probs + log_p_Oi

            class_prob[n, k] = P_ck * math.exp(sum_lognorm_probs)

    # make predictions
    for a in range(N):
        max_value = np.amax(class_prob[a, :])
        max_col_idx = np.where(class_prob[a, :] == max_value)[0]
        if max_col_idx.shape[0] == 2:
            class_pred[a, max_col_idx] = mate_row
        else:
            class_pred[a, max_col_idx] = 1

    correct_cnt = 0
    for b in range(N):
        if np.all(class_pred[b, :] == labels[b, :]):
            correct_cnt = correct_cnt + 1

    accuracy = correct_cnt * 1.0 / N

    print('Accuracy: {}'.format(accuracy))











