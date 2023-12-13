'''
This script computes the reference Sobol' indices values in the presence of hierarchical variables by relying
on a crude double Monte Carlo approach.

Please note that the execution of ths script may take several minutes, depending on the Monte Carlo sample size
'''

import numpy as np
import copy

def GSobol(X):
    x = X[:,:-1]
    Z =  X[:,-1]
    ai =  np.array([0,1,9,99,0,1,9,99])
    out = (np.abs(4*x-2)+ai)/(1+ai)
    out[np.where([Z == 1])[1],4:] = 1
    out = np.prod(out, axis = 1)
    
    return out

n = 10000  # Monte Carlo sample size
p = [0.5, 0.5]  # vector of : P(Z=1), P(Z=2)
z = [1, 2]  # different values of Z
np.random.seed(2345)
nb_variable = 8  # number of variables other than Z (value of d)

#First Monte Carlo sample
varval = np.hstack([np.random.uniform(size=(n, nb_variable)), np.random.choice(z, n, p=p).reshape(-1, 1)])
totvar = np.var(GSobol(varval))

# First order indices
first = []
mean = []

for nvar in range(nb_variable):
    mean = []

    for i in range(n):
        inp = copy.deepcopy(varval)
        inp[:,nvar] = np.random.uniform()
        # compute output
        y = GSobol(inp)
        mean.append(np.mean(y))
        if (i%10000 == 0):
            print(i)
    var = np.var(np.array(mean))/totvar
    first.append(var)
    
mean = []

for i in range(n):
    inp = copy.deepcopy(varval)
    inp[:,-1] = np.random.choice(z, 1, p=p)
    # compute output
    y = GSobol(inp)
    mean.append(np.mean(y))
    if (i%10000 == 0):
        print(i)
var = np.var(np.array(mean))/totvar
first.append(var)

# Second order indices
second = []
for nvar in range(nb_variable):
    mean = []

    for i in range(n):
        inp = copy.deepcopy(varval)
        inp[:,nvar] = np.random.uniform()
        inp[:,-1] = np.random.choice(z, 1, p=p)
        
        # compute output
        y = GSobol(inp)
        mean.append(np.mean(y))

    var = np.var(np.array(mean))/totvar
    second.append(var - first[nvar] - first[-1])

# Total order indices
tot = []

for nvar in range(nb_variable):
    mean = []

    for i in range(n):
        inp = np.array([np.hstack([np.random.uniform(size=(1, nb_variable)), np.random.choice(z, 1, p=p).reshape(-1, 1)])]*n)
        inp = inp.reshape((n,-1))
        inp[:,nvar] = np.random.uniform(size = n)
        # compute output
        y = GSobol(inp)
        mean.append(np.mean(y))

    var = np.var(np.array(mean))/totvar
    tot.append(1-var)

mean = []

for i in range(n):
    inp = np.array([np.hstack([np.random.uniform(size=(1, nb_variable)), np.random.choice(z, 1, p=p).reshape(-1, 1)])]*n)
    inp = inp.reshape((n,-1))
    inp[:,-1] = np.random.choice(z, n, p=p)
    # compute output
    y = GSobol(inp)
    mean.append(np.mean(y))
    if (i%10000 == 0):
        print(i)
var = np.var(np.array(mean))/totvar
tot.append(1-var)

