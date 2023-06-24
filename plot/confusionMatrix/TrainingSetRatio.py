
'''
RSM: Representation Similarity Matrix

RSM data is generated from the following code
https://colab.research.google.com/drive/1X_oghRsap7E2dO_qT1civMLqdD76y6nu?usp=sharing
'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib



########
dataset = 'cifar10'
# dataset = 'mnist'


net = 2
nets = [net] * 10
ratio = [i*0.1 for i in range(1, 11)]
n_task = 10


# ### load rsm from different training set ratio = 0.1, 0.2, ... ,1.0
arrays = []
vmin, vmax = 1, -1
for i in range(10):
    rsm = np.load('RSM/{}_net{}_ratio[{:.1f}].npy'.format(dataset, nets[i], ratio[i]))
    array = np.mean(rsm, axis=0)
    array = array[:n_task,:n_task]
    arrays.append(array)

    vmin = min(vmin, array.min())
    vmax = max(vmax, array.max())
print('vmax = {}\nvmin = {}'.format(vmax, vmin))


# ### load rsm from different runs, all of which use full dataset
baselines = []
vmin, vmax = 1, -1
for i in range(9):
    rsm = np.load('RSM/{}_net{}_rsm{}.npy'.format(dataset, nets[i], i + 1))
    baseline = np.mean(rsm, axis=0)
    baseline = baseline[:n_task,:n_task]
    baselines.append(baseline)


runs = 9
length = 10
distance_m = np.zeros((runs, length))
for run in range(runs):
    for j in range(length):
        distance_m[run][j] = np.linalg.norm(arrays[j] - baselines[run])

distance = np.min(distance_m, axis=0)
distance = distance/3



x = [i*0.1 for i in range(1, length+1)]
plt.plot(x, distance)
plt.xlabel('Ratio')
plt.ylabel('Matrix Distance')
plt.show()




