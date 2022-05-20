# before we do real experiments, we can plot a simulated figure to show how the execution order matters

import matplotlib.pyplot as plt
import numpy as np

class Simulation:

    def __init__(self):

        # suppose we have 10 tasks
        # [i, j, k] represents [state block id, encoder block id, classifier block id]
        self.Tasks = [[0, 0, 0],
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

        # number of tasks
        self.N = len(self.Tasks)

        # the ratio of the weight of each block
        # e.g. [0.5, 0.3, 0.2]
        self.Ratio = [[0.5, 0.3, 0.2], [0.5, 0.4, 0.1], [0.4, 0.4, 0.3]]

        # the model size of original end-to-end model
        self.Model_size = [100, 50, 20]  # end-to-end models may have different sizes, unit = KB

        # the overhead matrix
        self.Matrix = np.zeros((self.N, self.N))  # overhead matrix
        self.ComputeMatrix() # compute the overhead matrix


    def ComputeMatrix(self):

        # calculate size of each layer for each model size
        Model_size_perLayer = np.zeros((len(self.Model_size), len(self.Ratio[0])))  # weight size for each block/layer
        for i in range(len(self.Model_size)):
            for j in range(len(self.Ratio[0])):
                Model_size_perLayer[i][j] = self.Model_size[i] * self.Ratio[i][j]


        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                overhead = [Model_size_perLayer[self.Tasks[j][0]][k] for k in range(len(self.Tasks[i])) if
                            self.Tasks[i][k] != self.Tasks[j][k]]
                self.Matrix[i][j] = self.Matrix[j][i] = sum(overhead)


    def RunInstance(self, instance):
        # calculate the overhead of running one instance (one execution order)

        L = len(instance)  # length of the execution order
        Sum = 0
        for i in range(L - 1):  # only have (L - 1) times of switch
            Sum += self.Matrix[instance[i]][instance[i + 1]]  # add the overhead of each switch
        return Sum



    # this function was developed based on a function of printing permutations of a given list
    # reference: https://www.geeksforgeeks.org/generate-all-the-permutation-of-a-list-in-python/
    def permutation(self, lst, pre = -1):
        '''
        :param lst: a List lst, pre represents the taskID of previous consecutive task in the previous depth
                                if pre == '-1', it means we just enter the recursion loop, no recursion has happened yet
        :return: max and min value of all permutations of lst
        '''

        if pre == -1:
            if len(lst) == 1:
                print('Can not feed permutation(lst) with a list with only one element in the first call of this function!\n')

            if len(lst) > self.N:
                print('the length of the input list can not be larger than the number of total tasks')


        Max = float('-inf')
        Min = float('inf')

        # If there is only one element left in lst, it means we come to the end of the list
        # just return switch overhead switching from pre and lst[0]
        if len(lst) == 1:
            Max = Min = self.Matrix[pre][lst[0]]
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

            Max_temp, Min_temp = self.permutation(remLst, m)

            tempMaxList.append(Max_temp)
            tempMinList.append(Min_temp)

        if pre == -1:
            return max(tempMaxList), min(tempMinList)
        else:
            for i in range(len(tempMaxList)):
                Max = max(Max, self.Matrix[pre][lst[i]] + tempMaxList[i])
                Min = min(Min, self.Matrix[pre][lst[i]] + tempMinList[i])

        return Max, Min


    def runRandom(self, Repeat = 100, Plot = True):

        # we run a number of random list to see how the overall overhead changes
        self.Results = np.zeros((Repeat))
        for i in range(Repeat):
            Instance = np.arange(self.N)
            np.random.shuffle(Instance)
            self.Results[i] = self.RunInstance(Instance)

        print('Total {} tasks\n'.format(self.N))
        print('Random Best: {},\nRandom Wrost: {}\n'.format(min(self.Results), max(self.Results)))


        if Plot == True:

            # calculate the theory boundary by running all possible permutations
            theory_boundary = self.permutation(list(range(10)))
            print('Theory Best: {},\nTheory Wrost: {}\n'.format(theory_boundary[1], theory_boundary[0]))

            # figure drawing parameters
            fontsize = 13
            linewidth = 2
            fig, ax = plt.subplots()

            ax.plot(self.Results, 'b', label='random', linewidth=linewidth)
            ax.plot([theory_boundary[0]] * Repeat, 'k--', label='worst', linewidth=linewidth)
            ax.plot([theory_boundary[1]] * Repeat, 'r--', label='best', linewidth=linewidth)

            legend = ax.legend(loc='lower right', fontsize=fontsize - 4)

            plt.autoscale(enable=True, axis='x', tight=True)
            plt.xlabel('Repetition')
            plt.ylabel('Overall overhead of running all tasks once')

            fig.show()



















