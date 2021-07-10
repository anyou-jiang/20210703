import numpy as np
import math

def FitLinearGaussianParameters(X, U):
    '''
    :param X:
    :param U:
    :return:
    '''

    M = U.shape[0]
    N = U.shape[1]
    assert(N == 3)
    assert(M == X.shape[0])

    Beta = np.zeros((N+1, ))
    sigma = 1

    A = np.ones((N + 1, N + 1))
    mean_U = np.mean(U, axis=0)
    last_col = np.concatenate(([1], mean_U))
    first_row = np.concatenate((mean_U, [1]))
    A[0, :] = first_row
    A[:, -1] = last_col

    for i in range(N):
        for j in range(N):
            prod_ij = np.dot(U[:, i], U[:, j])
            E_ij = prod_ij / M
            A[i + 1, j] = E_ij

    B = np.ones((N + 1, ))
    first_row_B = np.mean(X)
    B[0] = first_row_B
    for k in range(N):
        prod_X_U_k = np.dot(X, U[:, k])
        E_X_U_k = prod_X_U_k / M
        B[k + 1] = E_X_U_k

    Beta = np.linalg.solve(A, B)

    cov_X_X = np.dot(X, X) / M - (np.mean(X)) ** 2
    cov_U = np.cov(U, rowvar=False, bias=True)

    beta_vec = Beta[0 : -1]
    beta_mat = np.matmul(np.matmul(beta_vec, cov_U), beta_vec.transpose())

    first_term = cov_X_X
    second_term = np.matmul(np.matmul(beta_vec, cov_U), beta_vec.transpose())
    sigma = math.sqrt(first_term - second_term)

    return Beta, sigma



    

