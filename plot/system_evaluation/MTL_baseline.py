"""
MTL baseline, we need to random generate the execution order to calculate its time/energy overhead
"""
import itertools

import numpy as np
import pandas as pd
import random

# To open Workbook
file = "comparison.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)

sheet_name_list = ['timeoverhead_msp','timeoverhead_pico','energyoverhead_msp','energyoverhead_pico']
sheet_name = sheet_name_list[3]  # change this index accordingly
df = xls.parse(sheet_name)  #
print(sheet_name)

names = df.values[0,1:10]  # name of datasets
values1 = df.values[10:16,1:10]  # layer-wise inference time or energy overhead
CostPerLayer_inference = np.transpose(values1)  # layer-wise inference time

values2 = df.values[20:26,1:10]  # layer-wise weight-reloading time or energy overhead
CostPerLayer_reload = np.transpose(values2) # layer-wise weights-reloading time

values3 = df.values[47:53,1:10]  # layer-wise weight-reloading time or energy overhead
MemoryPerLayer = np.transpose(values3) # layer-wise weights-reloading time


def get_CostPerBlock(CostPerLayer, BranchLoc):
    '''
    convert the layer-wise cost to block-wise cost according to the locations of branch out points
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


def get_CostPerBlock_new(CostPerLayer, BranchLoc):
    '''
    convert the layer-wise cost to block-wise cost according to the locations of branch out points
    :param CostPerLayer: an 2d array, layer-wise cost, row: dataset, col: layer
    :param BranchLoc: the index of branch locations
    :return: an 1d array, block-wise cost
    '''

    # n_row, _ = CostPerLayer.shape
    # CostPerBlock = np.zeros((4), dtype=object) # we only have three branch points, so we have 4 blocks in total

    # for i in range(n_row):
    CostPerBlock = [sum(CostPerLayer[:BranchLoc[0]+1]),           # block_0 - always shared by all tasks
                   sum(CostPerLayer[BranchLoc[0]+1:BranchLoc[1]+1]),   # block_1
                   sum(CostPerLayer[BranchLoc[1]+1:BranchLoc[2]+1]),   # block_2
                   sum(CostPerLayer[BranchLoc[2]+1:])]           # block_3

    return CostPerBlock


def cal_Matrix(N, decomposition):

    '''
    calculate a Matrix that shows the deepest shared block index among each task pair
    the cost of transferring from one task to another only depends on how deep the two tasks share blocks
    :param decomposition:
    :return:
    '''

    # decomposition = [   [[0, 1, 2, 3], [4]], [[0], [2], [1, 3], [4]]   ]
    # N = taskNum # number of tasks

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

def get_decomposition():

    ### we have to manually write the decomposition details of each dataset into the variable 'decompositions'
    d0 = [[[1], [0, 2, 3, 4, 5, 6, 7, 8, 9]], [[1], [4], [7], [0, 2, 3, 5, 6, 8, 9]]]  # MNIST
    d1 = [[[0], [2], [1, 3, 4, 5, 6, 7, 8, 9]], [[0], [2], [3], [1, 4, 5, 6, 7, 8, 9]]]  # CIFAR10
    d2 = [[[3], [8], [0, 1, 2, 4, 5, 6, 7, 9]], [[3], [8], [6], [0, 1, 2, 4, 5, 7, 9]]]  # SVHH
    d3 = [[[0], [8], [1, 2, 3, 4, 5, 6, 7, 9]], [[0], [8], [1, 2, 3, 4, 5, 6, 7], [9]]]  # GTSBR
    d4 = [[[3], [0, 1, 2, 4, 5, 6, 7], [8], [9]], [[3], [1], [0, 2, 4, 5, 6, 7], [8], [9]]]  # GSC
    d5 = [[[1], [2], [0, 3, 4, 5, 6, 7], [8], [9]], [[1], [2], [0, 3, 4, 5, 6, 7], [8], [9]]]  # FMNIST
    d6 = [[[1], [2], [5], [0, 3, 4, 6, 7, 8, 9]], [[1], [2], [5], [0, 3, 4, 6, 7, 8, 9]]]  # ESC
    d7 = [[[6, 8], [0, 1, 2, 3, 4, 5, 7, 9]], [[6], [8], [1], [4], [0, 2, 3, 5, 7, 9]]]  # US8K
    d8 = [[[1], [0, 2, 3, 4, 5]], [[1], [3], [0, 2, 4, 5]]]  # HHAR

    decompositions = [d0, d1, d2, d3, d4, d5, d6, d7, d8]

    return decompositions

def calc_MTL():
    '''
    all task pairs share two layers
    calculate the total execution overhead of running all tasks in MTL
    :return:
    '''

    print('\nMTL results:')
    for datasetIdx in range(9):

        #  datasetIdx - which dataset to use

        BranchLoc = [0,2,3]

        if datasetIdx == 8:
            N = 6
        else:
            N = 10


        CostPerBlock_inference = get_CostPerBlock(CostPerLayer_inference, BranchLoc)
        CostPerBlock_reload = get_CostPerBlock(CostPerLayer_reload, BranchLoc)

        cost_history = []
        order = [o for o in range(N)]
        order_ext = order + [order[0]]  # we need to append the first task to the end to make it a loop

        cost = 0
        for i in range(len(order_ext) - 1):
            SharedDepth = 0  # we assume for MTL, each task pair always only shares the first block of weights
            cost += sum(CostPerBlock_inference[datasetIdx][SharedDepth + 1:])
            cost += sum(CostPerBlock_reload[datasetIdx][SharedDepth + 1:])

        cost_history.append(cost)
        print(datasetIdx,cost)




def calc_SS_enumerateALL(datasetIdx = 8):

    '''
    Ideally, for smartswitch, we should enumerate all possible permutation, and calculate cost for each one
    and keep the lowest one
    :param datasetIdx:
    :return:
    '''
    decompositions = get_decomposition()

    # datasetIdx = 8 # which dataset to use

    # for datasets (Idx = 0,1,2,3,5,6) who have 6-layer, BranchLoc = [0,1,4], otherwise BranchLoc = [0,2,3] (Idx = 4,7,8)
    if datasetIdx in [0, 1, 2, 3, 5, 6]:
        BranchLoc = [0, 1, 4]
    else:
        BranchLoc = [0, 2, 3]

    # if datasetIdx = 8 (HHAR), N is 6, as HHAR has only 6 tasks
    if datasetIdx == 8:
        N = 6
    else:
        N = 10


    mat = cal_Matrix(N = N, decomposition = decompositions[datasetIdx]) # for SmartSwitch, calculate all datasets once, you need to write all decompositions into the variable 'decompositions'

    CostPerBlock_inference = get_CostPerBlock(CostPerLayer_inference, BranchLoc)
    CostPerBlock_reload = get_CostPerBlock(CostPerLayer_reload, BranchLoc)
    # order = [i for i in range(N)]

    cost_history = []
    iter = 0
    for order in itertools.permutations(list(range(N))):

        # random.shuffle(order)
        cost = 0
        transition = []

        order = list(order)
        order_ext = order + [order[0]]  #  we need to append

        for t_curr, t_next in zip(order_ext, order_ext[1:]):
            SharedDepth = mat[t_curr][t_next]
            transition.append(SharedDepth)
            cost += sum(CostPerBlock_inference[datasetIdx][SharedDepth+1:])
            cost += sum(CostPerBlock_reload[datasetIdx][SharedDepth+1:])

        cost_history.append(cost)
        if iter % 100000 == 0:
            print(iter, order, transition, cost)
        iter += 1

    print('datasetIdx = {}\nmin = {}'.format(datasetIdx, min(cost_history)))





def calc_SS():
    '''
    Ideally, for smartswitch, we should enumerate all possible permutation, and calculate cost for each one
    and keep the lowest one. However, our design does not have too many types of switch overhead (number of values in the overhead matrix)
    we can actually just randomly generate a few hundred samples and pick the lowest one
    '''


    print('\nSS results:')
    for datasetIdx in range(9):

        decompositions = get_decomposition()


        # for datasets (Idx = 0,1,2,3,5,6) who have 6-layer, BranchLoc = [0,1,4], otherwise BranchLoc = [0,2,3] (Idx = 4,7,8)
        if datasetIdx in [0, 1, 2, 3, 5, 6]:
            BranchLoc = [0, 1, 4]
        else:
            BranchLoc = [0, 2, 3]

        # if datasetIdx = 8 (HHAR), N is 6, as HHAR has only 6 tasks
        if datasetIdx == 8:
            N = 6
        else:
            N = 10


        mat = cal_Matrix(N = N, decomposition = decompositions[datasetIdx]) # for SmartSwitch, calculate all datasets once, you need to write all decompositions into the variable 'decompositions'

        CostPerBlock_inference = get_CostPerBlock(CostPerLayer_inference, BranchLoc)
        CostPerBlock_reload = get_CostPerBlock(CostPerLayer_reload, BranchLoc)



        cost_history = []
        order = list(range(N))


        for iter in range(100):

            random.shuffle(order)
            cost = 0
            transition = []

            order = list(order)
            order_ext = order + [order[0]]  #  we need to append

            for t_curr, t_next in zip(order_ext, order_ext[1:]):
                SharedDepth = mat[t_curr][t_next]
                transition.append(SharedDepth)
                cost += sum(CostPerBlock_inference[datasetIdx][SharedDepth+1:])
                cost += sum(CostPerBlock_reload[datasetIdx][SharedDepth+1:])

            cost_history.append(cost)
            # print(iter, order, transition, cost)

        print('{} - min = {}'.format(datasetIdx, min(cost_history)))



def calc_memory():
    '''
    We calculate the memory usage of our proposed method
    '''

    decompositions = get_decomposition()



    memory_total = 0
    for datasetIdx in range(9):




        # for datasets (Idx = 0,1,2,3,5,6) who have 6-layer, BranchLoc = [0,1,4], otherwise BranchLoc = [0,2,3] (Idx = 4,7,8)
        if datasetIdx in [0, 1, 2, 3, 5, 6]:
            BranchLoc = [0, 1, 4]
        else:
            BranchLoc = [0, 2, 3]

        # if datasetIdx = 8 (HHAR), N is 6, as HHAR has only 6 tasks
        if datasetIdx == 8:
            N = 6
        else:
            N = 10

        MemoryPerBlock = get_CostPerBlock_new(MemoryPerLayer[datasetIdx], BranchLoc)
        decomposition = decompositions[datasetIdx]

        memory = 0
        for i in range(4):
            if i == 0:  # the 1st block, all tasks share the 1st block
                memory += MemoryPerBlock[i]
            elif i == 3:  # the last block, all tasks have their own last block
                memory += MemoryPerBlock[i] * N
            else:  # the middle two blocks, it depends on the decomposition
                memory += MemoryPerBlock[i] * len(decomposition[i - 1])

        # for datasetIdx = 8 (HHAR), as HHAR has only 6 tasks, we need to time it with a multiplxier
        if datasetIdx == 8:
            memory /= (6/10)

        memory_total += memory
        print(memory)

    print('\nAveraged memory usage of 10 tasks \nin our proposed method is: \n{:.2f}KB'.format(memory_total/9))

calc_memory()


# calc_MTL()
# calc_SS()



