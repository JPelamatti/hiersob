import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


''' 
This script computes the first and total order Sobol' indices in the presence of hierarchical variables using knn estimators by relying
on an iid dataset

inputs:

    data: pandas dataframe, the hierarchical column must be labeled 'Z' and the output 'y'
    U: acceptable values are 0 and 1, U=0 results in the computation of total indices and U=1 of first order indices
    n_knn: integer, number of nearest neighbors used by the double MC estimator
    method: acceptable values are 0 and 1, the first option being for the double MC estimator, while the second one is for the P&F estimator
    
outputs:
    
    res: list containing 
    - if U = 1: the first order order indices associated to the inputs Z, X and the second order indices associated to the pairs X_i,Z
    - if U = 0: the total order order indices associated to the inputs Z and X

'''

def sobolpred(data, U=1, n_knn=2, method=1):
    X = data.drop(['Z', 'y'], axis=1)
    XZ =  data.drop(['y'], axis=1)
    Z = data['Z']
    y = data['y']
    
    N = len(y) if y is not None else data.shape[0]

    # Check that X, Z, and y have the same dimensionality
    if N != len(Z):
        raise ValueError("Z and y vectors must have the same length")
    elif X.shape[0] != N:
        raise ValueError("X and y must have the same dimensions")

    ####################################################
    # calculating conditional and non-conditional expectation and variance
    z = np.unique(Z)  # the different values that Z can take
    p = len(z)  # the number of different values that Z can take
    p_k = np.array([np.sum(Z == zi) / N for zi in z])
    e_k = np.array([np.mean(y[Z == zi]) for zi in z])
    v_k = np.array([np.var(y[Z == zi]) for zi in z])

    v_y = np.var(y)
    e_y = np.mean(y)

    ####################################################
    # Define the Ul list containing the subsets for which we want to compute the sobol index

    # first order indices
    if U == 1:
        Ul = [[u] for u in X.columns.tolist()]

    # total indices
    elif U == 0:
        Ul = [X.columns.difference([I]).tolist() for I in XZ.columns]

    else:
        raise ValueError("U is neither a list, nor equal to 0 or 1")

    L = len(Ul)

    ####################################################
    # Estimating S_z (first order)
    S_z = np.sum(p_k * (e_k * e_k - e_y ** 2)) / v_y

    ####################################################
    # Var(E[Y_k|X_I]) estimation by KNN,  for all k in [1,p] and I in Ul
    var_esp_cond_Yk = np.zeros((p, L))

    for k in range(p):
        # sub-sample where Z == z[k]
        X_k = X[Z == z[k]]
        y_k = y[Z == z[k]]

        for i in range(L):
            I = [el for el in Ul[i] if not np.isnan(X_k.iloc[0][el])]
            I_pow_k = [el for el in X_k.columns if not np.isnan(X_k.iloc[0][el])]

            if len(I) == 0:
                var_esp_cond_Yk[k, i] = 0
            elif len(I_pow_k) == len(I):
                var_esp_cond_Yk[k, i] = v_k[k]
            else:
                X_k_I = X_k[I]
                
                # Check if there is a large number of repetitions of the same values in the data set, which makes
                # the knn estimation provide wrong results
                if len(np.unique(X_k_I)) > 20:
                    # P&F estimation
                    if method == 1: 
                        knn = NearestNeighbors(n_neighbors=2)
                        knn.fit(X_k_I)
                        distances, indices = knn.kneighbors(X_k_I)
                        m = np.array(y_k.iloc[indices.reshape(-1, order = 'F')]).reshape((-1,2), order = 'F')
                        # prediction by pick & freeze of Var(E[Y_k|X_I])
                        var_esp_cond_Yk[k, i] = np.sum(m[:, 0] * m[:, 1]) / X_k_I.shape[0] - e_k[k] * e_k[k]
                        
                    # Double MC estimation
                    else:
                        knn = NearestNeighbors(n_neighbors=n_knn)
                        knn.fit(X_k_I)
                        distances, indices = knn.kneighbors(X_k_I)
                        m = np.array(y_k.iloc[indices.reshape(-1, order = 'F')]).reshape((-1,n_knn), order = 'F')
                        # prediction by double MC of Var(E[Y_k|X_I])
                        var_esp_cond_Yk[k, i] = v_k[k] - np.mean(np.var(m, axis=1, ddof = 1))
                        
                # In case of large number of repeated values, a proper conditioning estimation is performed
                else:
                    esp_cond_Yk = []
                    for un in np.unique(X_k_I):
                        esp_cond_Yk.append(np.mean(y_k.iloc[np.where(X_k_I == un)[0]]))
                    var_esp_cond_Yk[k, i] = np.var(np.array(esp_cond_Yk), ddof = 1)
                

    ####################################################
    # calculating sum_{k<l} { P(Z==z[k]) * P(Z==z[l]) * (E[E[Y_k|I]E[Y_l|I]] - E[Y_k]E[Y_l]) }
    # for all I in Ul

    big_sum = np.zeros(L)

    for i in range(L):
        s = 0

        for k in range(p - 1):
            for l in range(k + 1, p):
                # sub-sample where Z == z[k]
                X_k = X[Z == z[k]]
                y_k = np.array(y[Z == z[k]])

                # sub-sample where Z == z[l]
                X_l = X[Z == z[l]]
                y_l = np.array(y[Z == z[l]])

                I = [el for el in Ul[i] if not np.isnan(X_k.iloc[0][el] + X_l.iloc[0][el])]
                if len(I) != 0:

                    X_k = X_k[I]
                    X_l = X_l[I]
                    
                    # Check if there is a large number of repetitions of the same values in the data set, which makes
                    # the knn estimation provide wrong results
                    if len(np.unique(pd.concat([X_k,X_l]))) > 20:

    
                        knn = NearestNeighbors(n_neighbors=1)
                        knn.fit(X_l)
                        distances, indices = knn.kneighbors(X_k)
                        y_k_1nn = y_l[indices.ravel()]
    
                        knn = NearestNeighbors(n_neighbors=1)
                        knn.fit(X_k)
                        distances, indices = knn.kneighbors(X_l)
                        y_l_1nn = y_k[indices.ravel()]
    
                        s += (np.mean(np.append(y_k_1nn * y_k, y_l_1nn * y_l)) - e_k[k] * e_k[l]) * p_k[k] * p_k[l]
                        
                    # In case of large number of repeated values, a proper conditioning estimation is performed
                    else:
                        esp_cond_prod = []
                        for un in np.unique(pd.concat([X_k,X_l])):
                            nloc_k = len(np.where(X_k == un)[0])
                            nloc_l = len(np.where(X_l == un)[0])

                            [esp_cond_prod.append(np.mean(y_k[np.where(X_k == un)[0]])*np.mean(y_l[np.where(X_l == un)[0]])) for i in range(int((nloc_k+nloc_l)/2))]  

                        esp_cond_prod = np.mean(np.array(esp_cond_prod))
                        s += (esp_cond_prod - e_k[k] * e_k[l]) * p_k[k] * p_k[l]
                    big_sum[i] = s

    ####################################################
    # computing the indices

    # First-order indices:
    if U == 1:
        cols = X.columns.tolist()
        [cols.append('Z' + col) for col in X.columns]
        res = pd.DataFrame(columns=cols, index = range(1))
        for i in range(L):
            res.iloc[0][i] = (np.sum(p_k * p_k * var_esp_cond_Yk[:, i]) + 2 * big_sum[i]) / v_y
            res.iloc[0][i+L] = np.sum(p_k * var_esp_cond_Yk[:, i]) / v_y - res.iloc[0][i] 

        res['Z'] = S_z

    # Total indices:
    elif U == 0:
        res = pd.DataFrame(columns=range(L-1), index = range(1))
        S_z = 1 - (np.sum(p_k * p_k * var_esp_cond_Yk[:, -1]) + 2 * big_sum[-1]) / v_y
        for i in range(L - 1):
            res.iloc[0][i] = np.sum(p_k * (v_k - var_esp_cond_Yk[:, i])) / v_y

        res['Z'] = S_z



    return res