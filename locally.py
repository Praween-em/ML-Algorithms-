import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point, xmat, k): 
    m, n = np.shape(xmat)
    weights = np.mat(np.eye((m))) 
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k**2)) 
    return weights

def localWeight(point, xmat, ymat, k): 
    wei = kernel(point, xmat, k)
    W = (xmat.T * (wei * xmat)).I * (xmat.T * (wei * ymat.T)) 
    return W

def localWeightRegression(xmat, ymat, k): 
    m, n = np.shape(xmat)
    ypred = np.zeros(m) 
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k) 
    return ypred

data = pd.read_csv('lowweightregressiondataset.csv') 
bill = np.array(data.total_bill)
tip = np.array(data.tip)  # Separate line
bill = np.mat(bill) 
tip = np.mat(tip)  # Corrected variable name
m = np.shape(bill)[1]  # Corrected variable name
one = np.mat(np.ones(m))
x = np.hstack((one.T, bill.T))
ypred = localWeightRegression(x, tip, 3) 
SortIndex = x[:, 1].argsort(0)
xsort = x[SortIndex][:, 0]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
ax.scatter(np.array(bill), np.array(tip), color='green')
ax.plot(np.array(xsort[:, 1]), ypred[SortIndex], color='red', linewidth=3) 
plt.xlabel('Total Bill')
plt.ylabel('Tip') 
plt.show()
