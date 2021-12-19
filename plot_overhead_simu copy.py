# before we do real experiments, we can plot a simulated figure to show how the execution order matters

import matplotlib.pyplot as plt
import numpy as np

class Simulation:

    def __init__(self):

        self.tasks =




###############################  Functions   #############################################
##########################################################################################
# write a function to calculate the overall switch overhead of a given execution order
def Calculate(arr, matrix):
    L = len(arr)
    Sum = 0
    for i in range(L - 1):    # only have (L - 1) times of switch
        Sum += matrix[arr[i]][arr[i+1]] # add the overhead of each switch
        # print(matrix[i][i+1], Sum)
    return Sum

##########################################################################################
# Python function to print permutations of a given list
# reference: https://www.geeksforgeeks.org/generate-all-the-permutation-of-a-list-in-python/
def permutation(lst, pre = -1):
    '''
    :param lst: a List lst, pre represents the taskID of previous consecutive task in the previous depth
                            if pre == '-1', it means we just enter the recursion loop, no recursion has happened yet
    :return: max and min value of all permutations of lst
    '''

    Max = float('-inf')
    Min = float('inf')

    # return Max, Min

    # If there is only one element left in lst, it means we come to the end of the list
    # just return switch overhead switching from pre and lst[0]
    if len(lst) == 1:
        if pre == -1:
             print('Can not feed permutation(lst) with a list with only one element!\n')
        else:
            Max = Min = matrix[pre][lst[0]]
            return Max, Min


    # Iterate the input(lst) and calculate the permutation
    tempMaxList = []
    tempMinList = []
    for i in range(len(lst)):

        if pre == -1:
            print('finished {} out of {}'.format(i, len(lst)))

        m = lst[i]

        # Extract lst[i] or m from the list. remLst is
        # remaining list
        remLst = lst[:i] + lst[i+1:]

        # Generating all permutations where m is first
        # element
        # for Max_temp in permutation(remLst, m):

        Max_temp, Min_temp = permutation(remLst, m)

        tempMaxList.append(Max_temp)
        tempMinList.append(Min_temp)

    if pre == -1:
        return max(tempMaxList), min(tempMinList)
    else:
        for i in range(len(tempMaxList)):
            Max = max(Max, matrix[pre][lst[i]] + tempMaxList[i])
            Min = min(Min, matrix[pre][lst[i]] + tempMinList[i])

    return Max, Min
##########################################################################################
##########################################################################################


# suppose we have 10 tasks
# [i, j, k] represents [state block id, encoder block id, classifier block id]
tasks = [[0, 0, 0],
         [0, 0, 1],
         [0, 1, 2],
         # [0, 1, 10],
         # [0, 1, 11],

         [1, 2, 3],
         [1, 2, 4],
         [1, 3, 5],
         [1, 3, 6],
         # [1, 3, 12],

         [2, 4, 7],
         [2, 5, 8],
         [2, 5, 9],
         # [2, 6, 13],
         # [2, 6, 14]

         ]
N = len(tasks)  # the number of tasks


# the ratio of the weight of each block
# e.g. [0.5, 0.3, 0.2]
ratio = [[0.5, 0.3, 0.2], [0.5, 0.4, 0.1], [0.4, 0.4, 0.3]]

# the model size of original end-to-end model
Model_size = [100, 50, 20] # end-to-end models may have different sizes, unit = KB

# calculate size of each layer for each model size
Model_size_perLayer = np.zeros((len(Model_size), len(ratio[0]))) # weight size for each block/layer
for i in range(len(Model_size)):
    for j in range(len(ratio[0])):
        Model_size_perLayer[i][j] = Model_size[i] * ratio[i][j]



# calculate the overhead matrix
matrix = np.zeros((N, N))   # overhead matrix
for i in range(N - 1):
    for j in range(i + 1, N):
        overhead = [ Model_size_perLayer[tasks[j][0]][k] for k in range(len(tasks[i])) if tasks[i][k] != tasks[j][k] ]
        matrix[i][j] = matrix[j][i] = sum(overhead)

# calculate the theoretic best and worst overhead by trying every possible permutation
theory_boundary = permutation(list(range(10)))

# we run a number of random list to see how the overall overhead changes
Repeat = 400
Instances = np.zeros((Repeat))
for i in range(Repeat):
    arr = np.arange(N)
    np.random.shuffle(arr)
    # print(arr)
    Instances[i] = Calculate(arr, matrix)



print('Total {} tasks\n'.format(N))
print('Random Best: {},\nRandom Wrost: {}\n'.format(min(Instances), max(Instances)))
print('Theory Best: {},\nTheory Wrost: {}\n'.format(theory_boundary[1], theory_boundary[0]))



# plotting parameters
fontsize = 13
linewidth = 2
fig, ax = plt.subplots()

ax.plot(Instances, 'b', label = 'random', linewidth = linewidth)
ax.plot([theory_boundary[0]]*Repeat, 'k--', label = 'worst', linewidth = linewidth)
ax.plot([theory_boundary[1]]*Repeat, 'r--', label = 'best', linewidth = linewidth)

legend = ax.legend(loc='upper right', fontsize = fontsize - 4)

plt.autoscale(enable = True, axis = 'x', tight = True)
plt.xlabel('Repetition')
plt.ylabel('Overall overhead of running all tasks once')

fig.show()









