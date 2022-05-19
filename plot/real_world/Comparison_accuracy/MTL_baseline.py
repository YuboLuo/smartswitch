"""
MTL baseline, we need to random generate the execution order to calculate its time/energy overhead
"""

import numpy as np
import pandas as pd
import random

# To open Workbook
file = "comparison.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)
df = xls.parse('timeoverhead')

names = df.values[10,1:]  # name of datasets
values = df.values[11:17,1:]
CostPerLayer_inference = np.transpose(values)  # layer-wise inference time

values = df.values[21:27,1:]
CostPerLayer_reload = np.transpose(values) # layer-wise weights-reloading time


def get_CostPerBlock(CostPerLayer, BranchLoc):
    '''
    :param CostPerLayer: an 2d array, layer-wise cost, row: dataset, col: layer
    :param BranchLoc: the index of branch locations
    :return: an 1d array, block-wise cost
    '''

    n_row, _ = CostPerLayer.shape
    CostPerBlock = np.zeros((n_row, 4), dtype=object) # we only have three branch points, so we have 4 blocks in total

    for i in range(n_row):
        CostPerBlock[i] = [sum(CostPerLayer[i][:BranchLoc[0]+1]),           # block_0 - always shared by all tasks
                       sum(CostPerLayer[i][BranchLoc[0]+1:BranchLoc[1]+1]),   # block_1
                       sum(CostPerLayer[i][BranchLoc[1]+1:BranchLoc[2]+1]),   # block_2
                       sum(CostPerLayer[i][BranchLoc[2]+1:])]           # block_3

    return CostPerBlock


def cal_Matrix(decomposition = [   [[0, 1, 2, 3], [4]], [[0], [2], [1, 3], [4]]   ]):

    '''
    calculate a Matrix that shows the deepest shared block index among each task pair
    :param decomposition:
    :return:
    '''

    # decomposition = [   [[0, 1, 2, 3], [4]], [[0], [2], [1, 3], [4]]   ]
    N = 5 # number of tasks

    # # we use Matrix to show the deepest shared block index among each task pair
    Matrix = np.zeros((N, N), dtype=int)
    for i in range(N - 1):
        for j in range(i + 1, N):
            # # for each pair of tasks, we search them in the decomposition tree
            # # to see how deep they can go until they are branched out into different branches

            for idx, layer in enumerate(decomposition):
                for cluster in layer:
                    if i in cluster and j in cluster:

                        # # decomposition only contains the two middle layers decomposition details
                        # # so when idx = 0, it actually means they share up to the (idx + 1) layer
                        Matrix[i][j] = Matrix[j][i] = idx + 1
    return Matrix




decompositions = [ [[[0], [1, 2, 3], [4]], [[0], [1, 2, 3], [4]]], [[[1], [0, 2, 3, 4]], [[1], [3], [0, 2, 4]]], [[[3], [0, 1, 2, 4]], [[3], [0], [1, 2, 4]]], [[[3], [0, 1, 2, 4]], [[3], [0], [1, 2, 4]]], [[[3], [0, 1, 2, 4]], [[3], [1], [0, 2, 4]]]  ]


dataset = 0 # which dataset to use
BranchLoc = [0,2,4]  # for dataset = 4, BranchLoc = [0,2,3], otherwise BranchLoc = [0,2,4]

# mat = cal_Matrix(decomposition = decompositions[dataset]) # for SmartSwitch
mat = cal_Matrix(decomposition = [[[0],[1],[2],[3],[4]], [[0],[1],[2],[3],[4]]]) # for MTL

N = mat.shape[0]

CostPerBlock_inference = get_CostPerBlock(CostPerLayer_inference, BranchLoc)
CostPerBlock_reload = get_CostPerBlock(CostPerLayer_reload, BranchLoc)
order = [i for i in range(N)]

for iter in range(20):

    random.shuffle(order)
    cost, cost_history = 0, []
    transition = []

    order_ext = order + [order[0]]  #  we need to append

    # order_ext = order
    # cost += sum(CostPerBlock_inference[dataset])
    # cost += sum(CostPerBlock_reload[dataset])

    for t_curr, t_next in zip(order_ext, order_ext[1:]):
        SharedDepth = mat[t_curr][t_next]
        transition.append(SharedDepth)
        cost += sum(CostPerBlock_inference[dataset][SharedDepth+1:])
        cost += sum(CostPerBlock_reload[dataset][SharedDepth+1:])

    cost_history.append(cost)
    print(iter, order, transition, cost)

print(min(cost_history))



# cal_taskwise_overhead([[[0], [1, 2, 3], [4]], [[0], [1, 2, 3], [4]]]],)




