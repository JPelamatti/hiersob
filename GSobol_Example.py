'''This script computes the Sobol' indices values  in the presence of trigger variables for the modified
G-Sobol function'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sobolpred import sobolpred

plt.close('all')


def GSobol(X):
    out = np.ones(X.shape[0])
    for i in range(X.shape[0]):
        #Check hierarchical variable value
        if X[i,8] == 1:
            for d in range(4):
                ai = [0,1,9,99][d]
                out[i] *= (np.abs(4*X[i,d]-2)+ai)/(1+ai)
        if X[i,8] == 2:
            for d in range(8):
                ai = [0,1,9,99,0,1,9,99][d]
                out[i] *= (np.abs(4*X[i,d]-2)+ai)/(1+ai)
    return out

# parameterization
n = 10000  # sample size
p = [0.5, 0.5]  # vector containing: P(Z=1), P(Z=2)
z = [1, 2]  # different values of Z
nb = 20  # number of repetitions
alpha = 0.05  # confidence interval 95%
np.random.seed(2345) #For the sake of reproducibility
n_knn = 3 # nearest neighbors number

U = 1 #We start by computing the first order indices

# random sample generation & computation of indices
nb_variable = 8  # number of variables other than Z (d)

# matrix of Sobol indices of size: number of indices x nb for the Pick-Freeze and Monte Carlo variants
repetitions_PF = []
repetitions_MC = []

for i in range(nb):
    # sample generation
    m = np.hstack([np.random.uniform(size=(n, nb_variable)), np.random.choice(z, n, p=p).reshape(-1, 1)])
    
    # compute output
    y = GSobol(m)
    
    # convert to dataframe
    dataX = pd.DataFrame(m, columns=[f'X{i}' for i in range(1,nb_variable+1)] + ['Z'])
    # All inactive elements are replaced with nans
    dataX.loc[dataX['Z'] == 1, 'X5'] = np.nan
    dataX.loc[dataX['Z'] == 1, 'X6'] = np.nan
    dataX.loc[dataX['Z'] == 1, 'X7'] = np.nan
    dataX.loc[dataX['Z'] == 1, 'X8'] = np.nan
    dataX['y'] = y
    
    # Indices computation
    
    # Pick-Freeze Variant
    indices = sobolpred(dataX, U, n_knn, method = 1)
    repetitions_PF.append(indices)

    # Monte Carlo variant
    indices = sobolpred(dataX, U, n_knn, method = 2)
    repetitions_MC.append(indices)


repetitions_MC = np.array(repetitions_MC)
repetitions_MC = repetitions_MC.reshape((nb, -1))

repetitions_PF = np.array(repetitions_PF)
repetitions_PF = repetitions_PF.reshape((nb, -1))


# Plotting results, we have the first order indices associated to X and Z, as well 
# as the second order indices associated to the pairs X_i,Z
X = dataX.drop(['Z', 'y'], axis=1)
cols = X.columns.tolist()
cols = [cols[i][0]+'_'+ cols[i][1:] for i in range(len(cols))]
[cols.append('Z' + ',' + cols[i]) for i in range(len(cols))]
cols.append('Z')
cols = ['$S_{' + cols[i] + '}$' for i in range(len(cols))]

        
fig = plt.figure(figsize = (15,8))
bx1 = plt.boxplot(repetitions_MC, patch_artist = True,
            boxprops = dict(facecolor = "lightblue"),
            positions = np.arange(repetitions_MC.shape[1])-0.17,
            widths = 0.3)
bx2 = plt.boxplot(repetitions_PF, patch_artist = True,
            boxprops = dict(facecolor = "red"),
            positions = np.arange(repetitions_MC.shape[1])+0.17,
            widths = 0.3)


plt.xticks(np.arange(repetitions_MC.shape[1]), cols, fontsize = 16)
plt.legend([bx1["boxes"][0], bx2["boxes"][0]], ['MC KNN', 'PF KNN'], loc='best', fontsize = 16)
plt.show()
plt.ylabel('First order Sobol\' indices', fontsize = 16)
plt.tight_layout()
plt.savefig('GSobol_firstorder.pdf')



# Total order indices
U = 0

# matrix of Sobol indices of size: number of indices x nb for the Pick-Freeze and Monte Carlo variants
repetitions_PF = []
repetitions_MC = []

for i in range(nb):
    # sample generation
    m = np.hstack([np.random.uniform(size=(n, nb_variable)), np.random.choice(z, n, p=p).reshape(-1, 1)])
    
    # compute output
    y = GSobol(m)
    
    # convert to dataframe
    dataX = pd.DataFrame(m, columns=[f'X{i}' for i in range(1,nb_variable+1)] + ['Z'])
    dataX.loc[dataX['Z'] == 1, 'X5'] = np.nan
    dataX.loc[dataX['Z'] == 1, 'X6'] = np.nan
    dataX.loc[dataX['Z'] == 1, 'X7'] = np.nan
    dataX.loc[dataX['Z'] == 1, 'X8'] = np.nan
    dataX['y'] = y
    
    # Indices computation
    
    # Pick-Freeze Variant
    indices = sobolpred(dataX, U, n_knn, method = 1)
    repetitions_PF.append(indices)

    # Monte Carlo variant
    indices = sobolpred(dataX, U, n_knn, method = 2)
    repetitions_MC.append(indices)


repetitions_MC = np.array(repetitions_MC)
repetitions_MC = repetitions_MC.reshape((nb, -1))

repetitions_PF = np.array(repetitions_PF)
repetitions_PF = repetitions_PF.reshape((nb, -1))

# Plotting results, we have the total order indices associated to X and Z

X = dataX.drop(['Z', 'y'], axis=1)
cols = X.columns.tolist()
cols = [cols[i][0]+'_'+ cols[i][1:] for i in range(len(cols))]
cols.append('Z')
cols = ['$S_{' + cols[i] + '}$' for i in range(len(cols))]

fig = plt.figure(figsize = (15,8))
bx1 = plt.boxplot(repetitions_MC, patch_artist = True,
           boxprops = dict(facecolor = "lightblue"),
           positions = np.arange(repetitions_MC.shape[1])-0.17,
           widths = 0.3)
bx2 = plt.boxplot(repetitions_PF, patch_artist = True,
           boxprops = dict(facecolor = "red"),
           positions = np.arange(repetitions_MC.shape[1])+0.17,
           widths = 0.3)


plt.xticks(np.arange(repetitions_MC.shape[1]), cols, fontsize = 16)
plt.legend([bx1["boxes"][0], bx2["boxes"][0]], ['MC KNN', 'PF KNN'], loc='best', fontsize = 16)
plt.show()
plt.ylabel('Total order Sobol\' indices', fontsize = 16)
plt.tight_layout()
plt.savefig('GSobol_totalorder.pdf')
