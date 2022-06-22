"""
MTL baseline, we need to random generate the execution order to calculate its time/energy overhead
"""
import itertools

import numpy as np
import pandas as pd
import random

type = 1  # audio
# type = 0  # image

print('{}_based experiment'.format(['image','audio'][type]))

# To open Workbook
file = "comparison_wild.xlsx"

xls = pd.ExcelFile(file)
print(xls.sheet_names)


if type == 1:  # audio
    df = xls.parse('overhead_audio')
    values1 = df.values[7:12, 1:3]  # layer-wise inference time or energy overhead
    values2 = df.values[16:21,1:3]  # layer-wise weight-reloading time or energy overhead
else:  # image
    df = xls.parse('overhead_image')
    values1 = df.values[7:14, 1:3]  # layer-wise inference time or energy overhead
    values2 = df.values[18:25, 1:3]  # layer-wise weight-reloading time or energy overhead

CostPerLayer_inference = np.transpose(values1)  # layer-wise inference time
CostPerLayer_reload = np.transpose(values2) # layer-wise weights-reloading time


def get_CostPerBlock(CostPerLayer, BranchLoc):
    '''
    convert the layer-wise cost to block-wise cost according to the locations of branch out points
    :param CostPerLayer: an 2d array, layer-wise cost, row: dataset, col: layer
    :param BranchLoc: the index of branch locations
    :return: an 1d array, block-wise cost
    '''

    n_row, _ = CostPerLayer.shape
    CostPerBlock = np.zeros((n_row, len(BranchLoc) + 1), dtype=object) # we only have three branch points, so we have 4 blocks in total, len(BranchLoc) + 1) = 4

    for i in range(n_row):
        CostPerBlock[i] = [sum(CostPerLayer[i][:BranchLoc[0]+1]),           # block_0 - always shared by all tasks
                       sum(CostPerLayer[i][BranchLoc[0]+1:BranchLoc[1]+1]),   # block_1
                       sum(CostPerLayer[i][BranchLoc[1]+1:BranchLoc[2]+1]),   # block_2
                       sum(CostPerLayer[i][BranchLoc[2]+1:])]           # block_3

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

    if type == 1:  # audio
        d0 = [[[4], [0, 1, 2, 3]], [[4], [0, 1, 2], [3]]]  # audio_based experiment
        d1 = [[[4], [0, 1, 2, 3]], [[4], [0, 1, 2], [3]]]
    else:  # image
        d0 = [[[0], [1, 2, 3]], [[0], [1, 2, 3]]]
        d1 = [[[0], [1, 2, 3]], [[0], [1, 2, 3]]]

    decompositions = [d0, d1]

    return decompositions

def calc_MTL():
    '''
    all task pairs share two layers
    calculate the total execution overhead of running all tasks in MTL
    :return:
    '''

    overheadtype = ['time', 'energy']
    print('\nMTL results:')
    for Idx in range(2):

        #  datasetIdx - which dataset to use



        if type == 1:
            N = 5  # audio_based experiment has 5 tasks
            BranchLoc = [0, 2, 3]  # audio_based experiment has 5 layers
        else:
            N = 4  # image_based experiment has 4 tasks
            BranchLoc = [0, 4, 5]  # image_based experiment has 7 layers


        CostPerBlock_inference = get_CostPerBlock(CostPerLayer_inference, BranchLoc)
        CostPerBlock_reload = get_CostPerBlock(CostPerLayer_reload, BranchLoc)

        cost_history = []
        order = [o for o in range(N)]
        order_ext = order + [order[0]]  # we need to append the first task to the end to make it a loop

        cost = 0
        for i in range(len(order_ext) - 1):
            SharedDepth = 0  # we assume for MTL, each task pair always only shares the first block of weights
            cost += sum(CostPerBlock_inference[Idx][SharedDepth + 1:])
            cost += sum(CostPerBlock_reload[Idx][SharedDepth + 1:])

        cost_history.append(cost)
        print(Idx,overheadtype[Idx],cost)





def calc_SS():
    '''
    Ideally, for smartswitch, we should enumerate all possible permutation, and calculate cost for each one
    and keep the lowest one. However, our design does not have too many types of switch overhead (number of values in the overhead matrix)
    we can actually just randomly generate a few hundred samples and pick the lowest one
    '''

    overheadtype = ['time', 'energy']
    print('\nSS results:')
    for Idx in range(2):

        decompositions = get_decomposition()


        if type == 1:
            N = 5  # audio_based experiment has 5 tasks
            BranchLoc = [0, 2, 3]  # audio_based experiment has 5 layers
        else:
            N = 4  # image_based experiment has 4 tasks
            BranchLoc = [0, 4, 5]  # image_based experiment has 7 layers


        mat = cal_Matrix(N = N, decomposition = decompositions[Idx]) # for SmartSwitch, calculate all datasets once, you need to write all decompositions into the variable 'decompositions'

        CostPerBlock_inference = get_CostPerBlock(CostPerLayer_inference, BranchLoc)
        CostPerBlock_reload = get_CostPerBlock(CostPerLayer_reload, BranchLoc)
        # order = [i for i in range(N)]

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
                cost += sum(CostPerBlock_inference[Idx][SharedDepth+1:])
                cost += sum(CostPerBlock_reload[Idx][SharedDepth+1:])

            cost_history.append(cost)
            # print(iter, order, transition, cost)

        print('{} - {} - min = {}'.format(Idx, overheadtype[Idx], min(cost_history)))


def calc_SS_TSPPC():
    '''
    calculate the overhead results for TSPPC version
    we impose a precedence constraint which requires task_0 (presence detection) be done before all other tasks
    '''

    overheadtype = ['time', 'energy']
    print('\nSS TSPPC results:')
    for Idx in range(2):

        decompositions = get_decomposition()


        if type == 1:
            N = 5  # audio_based experiment has 5 tasks
            BranchLoc = [0, 2, 3]  # audio_based experiment has 5 layers
        else:
            N = 4  # image_based experiment has 4 tasks
            BranchLoc = [0, 4, 5]  # image_based experiment has 7 layers


        mat = cal_Matrix(N = N, decomposition = decompositions[Idx]) # for SmartSwitch, calculate all datasets once, you need to write all decompositions into the variable 'decompositions'

        CostPerBlock_inference = get_CostPerBlock(CostPerLayer_inference, BranchLoc)
        CostPerBlock_reload = get_CostPerBlock(CostPerLayer_reload, BranchLoc)
        # order = [i for i in range(N)]

        cost_history = []
        order = list(range(N))

        for iter in range(100):

            random.shuffle(order)
            cost = 0
            transition = []

            order = list(order)

            # swap task_0 to the first task in the order as task_0 is the precedence constraint of all other tasks
            order[order.index(0)], order[0] = order[0], order[order.index(0)]

            order_ext = order + [order[0]]  #  we need to append

            for t_curr, t_next in zip(order_ext, order_ext[1:]):
                SharedDepth = mat[t_curr][t_next]
                transition.append(SharedDepth)
                cost += sum(CostPerBlock_inference[Idx][SharedDepth+1:])
                cost += sum(CostPerBlock_reload[Idx][SharedDepth+1:])

            cost_history.append(cost)
            # print(iter, order, transition, cost)

        print('{} - {} - min = {}'.format(Idx, overheadtype[Idx], min(cost_history)))



def calc_SS_TSPCC():
    '''
    calculate the overhead results for TSPCC version
    we impose a precedence constraint which requires task_0 (presence detection) be done before all other tasks
    '''

    overheadtype = ['time', 'energy']
    print('\nSS TSPCC results:')
    for Idx in range(2):

        decompositions = get_decomposition()


        if type == 1:
            N = 5  # audio_based experiment has 5 tasks
            BranchLoc = [0, 2, 3]  # audio_based experiment has 5 layers
        else:
            N = 4  # image_based experiment has 4 tasks
            BranchLoc = [0, 4, 5]  # image_based experiment has 7 layers


        mat = cal_Matrix(N = N, decomposition = decompositions[Idx]) # for SmartSwitch, calculate all datasets once, you need to write all decompositions into the variable 'decompositions'

        CostPerBlock_inference = get_CostPerBlock(CostPerLayer_inference, BranchLoc)
        CostPerBlock_reload = get_CostPerBlock(CostPerLayer_reload, BranchLoc)
        # order = [i for i in range(N)]

        cost_history = []
        order = list(range(N))

        for iter in range(100):

            random.shuffle(order)
            cost_execute, cost_skip = 0, 0
            transition = []

            order = list(order)

            # swap task_0 to the first task in the order as task_0 is the precedence constraint of all other tasks
            order[order.index(0)], order[0] = order[0], order[order.index(0)]

            order_ext = order + [order[0]]  #  we need to append


            for t_curr, t_next in zip(order_ext, order_ext[1:]):
                SharedDepth = mat[t_curr][t_next]
                transition.append(SharedDepth)
                cost_execute += sum(CostPerBlock_inference[Idx][SharedDepth+1:])
                cost_execute += sum(CostPerBlock_reload[Idx][SharedDepth+1:])

            cost_skip = sum(CostPerBlock_inference[Idx])  # for cost_skip case, we only execute task_0 once
            cost = cost_skip * 0.8 + cost_execute * 0.2

            cost_history.append(cost)
            print(iter, order, transition, cost)

        print('{} - {} - min = {}'.format(Idx, overheadtype[Idx], min(cost_history)))





calc_MTL()
calc_SS()
calc_SS_TSPPC()
calc_SS_TSPCC()


