import tensorflow as tf

print('tensorflow version:', tf.__version__)  # we use tensorflow 2.x

from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras import backend as F
from tensorflow import keras
from scipy import stats
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import itertools, more_itertools
import time

def dataload_mnist(chosenType):
    '''
    divide the original MNIST dataset into single digit recognition dataset (a binary classification problem)
    with the chosenType = 1, all other types = 0
    :param chosenType: the class type you pick as the 1 (other types will be 0)
    :return: training and testing dataset
    '''

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    ### Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    ### make the label list binary with only chosenType being 1 and all other types being 0
    y_test = [1 if c == chosenType else 0 for c in y_test]
    y_train = [1 if c == chosenType else 0 for c in y_train]

    ### expand dimension
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    y_train = tf.expand_dims(y_train, -1)
    y_test = tf.expand_dims(y_test, -1)

    return x_train, x_test, y_train, y_test


def model_train(train=True, chosenType=0):
    '''
    Retrain a new model or load a pretrained model
    :param train: True means to retrain a model; False means to load a pretrained model
    :param chosenType: the class type you use as 1
    :return: return the keras model
    '''

    if not train:  # load model data from pretrained model
        model = keras.models.load_model('pretrained/mnist/mnist_{}'.format(chosenType))

    else:  # else build and train a new model, and save it

        x_train, x_test, y_train, y_test = dataload_mnist(chosenType)

        model = models.Sequential()

        model.add(layers.Conv2D(8, (7, 7), strides=(2, 2), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.Conv2D(8, (2, 2), strides=(2, 2), activation='relu', input_shape=(11, 11, 8)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2))
        # model.summary()

        # compile and train
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

        # save model
        model.save('pretrained/mnist/mnist_{}'.format(chosenType))
        print('Model saved: pretrained/mnist/mnist_{}'.format(chosenType))

    return model


def RDM_Calc(K, chosenType=0, train=False, getFuncNumber=False):
    '''
    calculate the Representation Dissimilarity Matrix (RDM) for each possible branch out point
    this function is for one task, you need to run this function repeatedly for all tasks
    :param K: the number of images to use for calculation, better to use 50 or more
    :param chosenType: the class type you use as 1 for this task
    :param getFuncNumber: if this param is True, we just return the number of func, a.k.a the number of branch out points
    :return: RDM for this task
    '''

    model = model_train(chosenType=chosenType, train=train)
    x_train, x_test, _, _ = dataload_mnist(chosenType)
    images = x_test[:K]

    # get the name of each layers for debug
    layersName = {idx: [layer.name, layer.output.shape] for idx, layer in enumerate(model.layers)}

    '''
    setup functions to read outputs of intermediate layers
    those functions are related to specific model architecture, so design here accordingly
    '''
    func0 = F.function([model.layers[0].input], [model.layers[1].output])
    func1 = F.function([model.layers[0].input], [model.layers[4].output])
    func2 = F.function([model.layers[0].input], [model.layers[5].output])
    funcList = [func0, func1, func2]

    if getFuncNumber:
        return len(funcList)

    ### read outputs from intermediate layers of the model using K images
    K = len(images)  # K images were used
    outs = []
    for func in funcList:
        out = func(images)[0]

        ### the 3-D tensors are linearized to 1-D tensors
        outs.append(out.reshape(out.shape[0], int(out.size / out.shape[0])))

    ### after we get intermediate results, we use them to calculate the Representation Dissimilarity Matrix (RDM)
    RDM = np.zeros((len(outs), K, K))
    for idx, out in enumerate(outs):
        ### for each func (a.k.a each branch out point)
        for i in range(K):
            for j in range(i, K):
                ### pearson correlation coefficient tells you the correlation, we need the dissimilarity, so we use (1 - p)
                RDM[idx][i][j] = 1 - stats.pearsonr(out[i], out[j])[0]
                RDM[idx][j][i] = RDM[idx][i][j]  # the matrix is symmetric

    return RDM


def RSM_Calc(K):
    '''
    calculate the task-wise Representation Similarity Matrix (RSM) - the paper calls it task affinity tensor A
    :param K: the number of images to use for RDM calculation
    :return:
    '''

    # some parameters
    T = 10  # number of tasks, mnist dataset has 10 classes and we divide it into 10 individual tasks
    D = RDM_Calc(1, getFuncNumber=True)  # number of division points where you possibly branch out
    RSM = np.zeros((D, T, T))

    # calculate RDM for each task
    RDM = [RDM_Calc(K, chosenType=t) for t in range(T)]

    ###
    for d in range(D):
        for i in range(T):
            for j in range(T):
                # extract RDM of the d_th division point for task_i and task_j
                m1, m2 = RDM[i][d], RDM[j][d]

                # extract the upper triangle of the matrix and flatten them into a list
                p1 = [elem for ii, row in enumerate(m1) for jj, elem in enumerate(row) if ii < jj]
                p2 = [elem for ii, row in enumerate(m2) for jj, elem in enumerate(row) if ii < jj]

                # calculate the Spearman’s correlation coefficient for task_i and task_j at the d_th division point
                RSM[d][i][j] = stats.spearmanr(p1, p2).correlation
    return RSM


def partition(collection):
    '''
    Bell Number reference: https://en.m.wikipedia.org/wiki/Bell_number
    implementation reference: https://stackoverflow.com/questions/19368375/set-partitions-in-python
    return all possible partitions of a list of unique numbers
    "A partition of a set X is a set of non-empty subsets of X such that
    every element x in X is in exactly one of these subsets"
    :param collection: a list of unique numbers
    :return: all possible partitions
    '''
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def ScoreCalc(clusters, depth, RSM):
    '''
    calculate the task dissimilarity score for a given division point
    :param clusters: the input cluster list, e.g. for [[1,2],[3,4]], [1,2] is a cluster, [3,4] is a cluster
    :param depth: the division point
    :param RSM: pre-computed Representation Similarity Matrix - the paper calls it task affinity tensor A
    :return: the averaged maximum distance between the dissimilarity scores of the elements in every cluster
    '''

    tobeAveraged = 0
    for cluster in clusters:
        n = len(cluster)

        if n <= 1:
            tobeAveraged += 0  # the dissimilarity between a task and itself is 0
        else:
            '''
            calculate all pair-wise dissimilarity scores for all elements in this cluster
            then only keep the largest score and add it to the tobeAveraged variable
            '''
            max_score = float('-inf')
            for i in range(n - 1):
                for j in range(i + 1, n):

                    # 1 - RSM(d, i, j) is the dissimilarity score between task_i and task_j at division point d
                    if 1 - RSM[depth][cluster[i]][cluster[j]] > max_score:
                        max_score = 1 - RSM[depth][cluster[i]][cluster[j]]

            tobeAveraged += max_score

    return tobeAveraged / len(clusters)


def weak_compositions(boxes, balls, parent=tuple(), depth = 0, constraint = [float('inf')] * 30):
    '''
    when we branch out from layer L to layer L + 1, for the partition at layer L, it is easy
    you can just use more_itertools.set_partitions which allows you to specify the number of partitions you want
    however, for layer L + 1, it is not trivial to do, let's use an example to analysis

    Example: assume at layer L, the partition result is [[1,2],[3],[4,5,6]] and we have 4 partitions at layer L + 1
        so at layer L, we have 3 partitions in [[1,2],[3],[4,5,6]], in order to branch into a 4-partition result at layer L + 1
        we can divide the [1,2] so we have [[1],[2],[3],[4,5,6]] at layer L + 1, or we can divide [4,5,6] and have [[1,2],[3],[4],[5,6]]
        and there are many other ways to divide [4,5,6] into two parts.

        In our solution, we can covert this into a "toss N balls into M boxes" problem
        the number of balls (N) is the number of partitions at layer L + 1
        the number of boxes (M) is the number of partitions at layer L
        in our example, N = 4, M = 3
        then valid tossing strategy includes: [4, 0, 0], [3, 1, 0], [1, 2, 1] ...
        values in each tossing strategy represents that number of parts we should divide the corresponding partition in layer L
        e.g. [1, 2, 1] means that
                            we divide the 1st part - [1,2] of layer L into 1 part
                            we divide the 2nd part - [3] of layer L into 2 parts  (does not make sense)
                            we divide the 3rd part - [4,5,6] of layer L into 1 part (does not make sense)
        As you can see here, the strategy may have have non-sense for some divisions. We need to add constraints
        for the general "toss N balls into M boxes" problem
        Constraints: (1) each box must have at least one ball - here means each part of layer L if not divided should be at least kept as it is
                     (2) the number of balls in a box should be smaller than the number of values in its corresponding part at layer L
                            e.g. you can not divide [1,2] into three parts

        Final solution, we borrow the function wrote in this post https://stackoverflow.com/questions/4647120/next-composition-of-n-into-k-parts-does-anyone-have-a-working-algorithm
            go to the above link and ctr+F to search "weak_compositions" and you will find the code we borrowed
            we modify the code by eliminating (1) any 0 value, and (2) any value that exceed its corresponding constraint value

    :param depth: depth of the recursion, we use this to index the value of constraint for later comparsion
    :param constraint: the number of values in each part of layer L
                        e.g. [[1,2],[3],[4,5,6]] at layer L, the constraint should be [2,1,3]
                       the length of constraint should be larger than the number of parts at layer L, 30 should be big enough for our case

    '''

    max_depth = len(constraint)

    if boxes > 1:
        for i in range(1, balls):  # modified the original range(balls + 1), to eliminate 0 value

            if balls - i > constraint[depth]: # apply constraint for all non-final depth in recursion
                continue

            if depth == max_depth - 2 and i > constraint[-1]: # apply constraint for the final depth in recursion
                continue

            for x in weak_compositions(boxes - 1, i, parent + (balls - i,), depth + 1, constraint=constraint):
                yield x
    else:
        yield parent + (balls,)


def clustering_withBudget(RSM, N=5, Budget=5):
    '''
    enumerate all possible trees under a budget
    we only enumerate trees that use the maximum possible budget which is no greater than the given input Budget
    :param RSM: pre-computed Representation Similarity Matrix - the paper calls it task affinity tensor A
    :param N: total number of tasks
    :param Budget: the number of nodes on top of the minimal budget
    '''

    # N = 9 # number of tasks
    tasks = list(range(N))
    queue = []  # to save all possible trees, and their score and model size

    # get the weight size of each layer
    model = model_train(train=False)
    var = model.trainable_variables
    sizeByLayer = [F.count_params(i) + F.count_params(j) for i, j in zip(var[0::2], var[1::2])]
    sizeByBlock = [sizeByLayer[0] + sizeByLayer[1]] + sizeByLayer[2:]

    '''
    our structure only has 4 layers of blocks:
        0th layer: only 1 node
        1st layer: flexible, to be determined
        2nd layer: flexible, to be determined
        3th layer: len(task) nodes because you must branch into individual task at this layer
    for a given budget, we enumerate all possible arrangements for the 1st & 2nd layers

    Constraints: 
        the budget nodes of the 1st layer should be no greater than that of the 2nd layer
        the budget nodes of all layers should be less than the number of tasks
    '''
    arrangement = []
    for i in range(1, int(Budget / 2) + 1):
        if i <= N and Budget - i <= N:
            arrangement.append([i, Budget - i])

    cnt1 = cnt2 = 0
    for arrange in arrangement:
        for clusters0 in more_itertools.set_partitions(tasks, k=arrange[0]):  # arrange[0]: the number of nodes for 1st layer

            ''' Variable Name Explanation 
            clustersX, depthX, scoreX: 'X' means the Xth branch out point
            clustersX: means the branched-out result at Xth branch out point
            scoreX: means the dissimilarity score for clustersX
            '''

            # continue

            depth0, depth1 = 0, 1
            score0 = ScoreCalc(clusters0, depth0, RSM)

            # print(clusters0)
            # print('--->')

            for decompose in weak_compositions(arrange[0], arrange[1], constraint=[len(p) for p in clusters0]):

                # for clusters1 in itertools.product(*[partition(cluster) for cluster in clusters0]):
                for clusters1 in itertools.product(
                        *[more_itertools.set_partitions(cluster, k=k) for cluster, k in zip(clusters0, decompose)]):

                    '''
                    we use itertools.product to output all combinations of possible subtrees
                    please refer to: https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
                    to understand how itertools.product(*[ ]) works
                    '''

                    # strip the out-layer bracket
                    res = []
                    for l in clusters1:
                        res += l
                    clusters1 = res
                    # print(res)

                    if len(clusters1) != arrange[1]:
                        cnt1 += 1  # unnecessary searches
                        continue

                    cnt2 += 1
                    score1 = ScoreCalc(clusters1, depth1, RSM)

                    '''
                    our paper only has three layers of branch out points, for the final one, it is always branched into single tasks
                    so, we do not need to continue the processing of clusters2/score2
                    clusters2 will all be single tasks, e.g. clusters2 should always be [[0],[1],[2],[3],[4]] if 5 tasks in total. 
                    then score2 will always be 0; 
                    '''

                    # summarize all info and pack them into the queue
                    score = score0 + score1  # total dissimilarity score of all branch out points, score2 is omitted as it is 0
                    model_size = sizeByBlock[0] + len(clusters0) * sizeByBlock[1] \
                                 + len(clusters1) * sizeByBlock[2] \
                                 + len(tasks) * sizeByBlock[3]  # total model size this tree requires
                    # model_size = sizeByBlock[0] + sizeByBlock[1] + sizeByBlock[2] + sizeByBlock[3]
                    tree = [clusters0, '--->', clusters1]  # decomposition details

                    queue.append([score, model_size, tree])

                # print('\n\n')

            queue.sort(key=lambda x: (x[1], x[0]))  # first sort by model_size, then by score
    print('cnt1 = {}, cnt2 = {}, unnecessary searches ratio = {:.3}'.format(cnt1, cnt2, cnt1 / (cnt1 + cnt2)))
    return queue


def clustering_withBudget_old(RSM, N = 5, Budget = 5):
    '''
    enumerate all possible trees under a budget
    we only enumerate trees that use the maximum possible budget which is no greater than the given input Budget
    :param RSM: pre-computed Representation Similarity Matrix - the paper calls it task affinity tensor A
    :param N: total number of tasks
    :param Budget: the number of nodes on top of the minimal budget
    '''

    # N = 9 # number of tasks
    tasks = list(range(N))
    queue = []  # to save all possible trees, and their score and model size

    # get the weight size of each layer
    model = model_train(train=False)
    var = model.trainable_variables
    sizeByLayer = [F.count_params(i) + F.count_params(j) for i, j in zip(var[0::2], var[1::2])]
    sizeByBlock = [sizeByLayer[0] + sizeByLayer[1]] + sizeByLayer[2:]

    '''
    our structure only has 4 layers of blocks:
        0th layer: only 1 node
        1st layer: flexible, to be determined
        2nd layer: flexible, to be determined
        3th layer: len(task) nodes because you must branch into individual task at this layer
    for a given budget, we enumerate all possible arrangements for the 1st & 2nd layers
    
    Constraints: 
        the budget nodes of the 1st layer should be no greater than that of the 2nd layer
        the budget nodes of all layers should be less than the number of tasks
    '''
    arrangement = []
    for i in range(1, int(Budget / 2) + 1):
        if i <= N and Budget - i <= N:
            arrangement.append([i, Budget - i])

    cnt1 = cnt2 = 0
    for arrange in arrangement:
        for clusters0 in more_itertools.set_partitions(tasks, k = arrange[0]):  # arrange[0]: the number of nodes for 1st layer

            ''' Variable Name Explanation 
            clustersX, depthX, scoreX: 'X' means the Xth branch out point
            clustersX: means the branched-out result at Xth branch out point
            scoreX: means the dissimilarity score for clustersX
            '''

            # continue

            depth0, depth1 = 0, 1
            score0 = ScoreCalc(clusters0, depth0, RSM)

            # print(clusters0)
            # print('--->')


            for clusters1 in itertools.product(*[partition(cluster) for cluster in clusters0]):
            # for clusters1 in itertools.product(*[more_itertools.set_partitions(cluster, k = k) for cluster, k in zip(clusters0, decompose)]):



                '''
                we use itertools.product to output all combinations of possible subtrees
                please refer to: https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
                to understand how itertools.product(*[ ]) works
                '''

                # strip the out-layer bracket
                res = []
                for l in clusters1:
                    res += l
                clusters1 = res
                # print(res)

                if len(clusters1) != arrange[1]:
                    cnt1 += 1  # unnecessary searches
                    continue

                cnt2 += 1
                score1 = ScoreCalc(clusters1, depth1, RSM)

                '''
                our paper only has three layers of branch out points, for the final one, it is always branched into single tasks
                so, we do not need to continue the processing of clusters2/score2
                clusters2 will all be single tasks, e.g. clusters2 should always be [[0],[1],[2],[3],[4]] if 5 tasks in total. 
                then score2 will always be 0; 
                '''

                # summarize all info and pack them into the queue
                score = score0 + score1  # total dissimilarity score of all branch out points, score2 is omitted as it is 0
                model_size = sizeByBlock[0] + len(clusters0) * sizeByBlock[1] \
                             + len(clusters1) * sizeByBlock[2] \
                             + len(tasks) * sizeByBlock[3]  # total model size this tree requires
                # model_size = sizeByBlock[0] + sizeByBlock[1] + sizeByBlock[2] + sizeByBlock[3]
                tree = [clusters0, '--->', clusters1]  # decomposition details

                queue.append([score, model_size, tree])

            # print('\n\n')

            queue.sort(key=lambda x: (x[1], x[0])) # first sort by model_size, then by score
    print('cnt1 = {}, cnt2 = {}, unnecessary searches ratio = {:.3}'.format(cnt1, cnt2, cnt1/(cnt1 + cnt2) ))
    return queue

def clustering(RSM, N = 5):
    '''
    enumerate all possible trees and calculate their dissimilarity score
    :param RSM: pre-computed Representation Similarity Matrix - the paper calls it task affinity tensor A
    :param N: total number of tasks
    '''

    # N = 9 # number of tasks
    tasks = list(range(N))
    queue = []  # to save all possible trees, and their score and model size

    # get the weight size of each layer
    model = model_train(train=False)
    var = model.trainable_variables
    sizeByLayer = [F.count_params(i) + F.count_params(j) for i, j in zip(var[0::2], var[1::2])]
    sizeByBlock = [sizeByLayer[0] + sizeByLayer[1]] + sizeByLayer[2:]

    for clusters0 in partition(tasks):

        ''' Variable Name Explanation 
        clustersX, depthX, scoreX: 'X' means the Xth branch out point
        clustersX: means the branched-out result at Xth branch out point
        scoreX: means the dissimilarity score for clustersX
        '''

        depth0, depth1 = 0, 1
        score0 = ScoreCalc(clusters0, depth0, RSM)

        # print(clusters0)
        # print('--->')
        for clusters1 in itertools.product(*[partition(cluster) for cluster in clusters0]):
            '''
            we use itertools.product to output all combinations of possible subtrees
            please refer to: https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
            to understand how itertools.product(*[ ]) works
            '''

            # strip the out-layer bracket
            res = []
            for l in clusters1:
                res += l
            clusters1 = res
            # print(res)

            score1 = ScoreCalc(clusters1, depth1, RSM)

            '''
            our paper only has three layers of branch out points, for the final one, it is always branched into single tasks
            so, we do not need to continue the processing of clusters2/score2
            clusters2 will all be single tasks, e.g. clusters2 should always be [[0],[1],[2],[3],[4]] if 5 tasks in total. 
            then score2 will always be 0; 
            '''

            # summarize all info and pack them into the queue
            score = score0 + score1  # total dissimilarity score of all branch out points, score2 is omitted as it is 0
            model_size = sizeByBlock[0] + len(clusters0) * sizeByBlock[1] \
                         + len(clusters1) * sizeByBlock[2] \
                         + len(tasks) * sizeByBlock[3]  # total model size this tree requires
            tree = [clusters0, '--->', clusters1]  # decomposition details

            queue.append([score, model_size, tree])

        # print('\n\n')

        queue.sort(key=lambda x: (x[1], x[0])) # first sort by model_size, then by score

    return queue

def plotQueue(queue):
    '''
    Plot how dissimilarity scores vary among all possible budgets
    :param queue:
    :return:
    '''
    dic = defaultdict(list)
    for q in queue:
        dic[q[1]].append(q[0])

    for idx, key in enumerate(dic.keys()):
        plt.boxplot(dic[key], positions=[idx], showfliers=False)  # do not plot outliers

    plt.show()


def optimalTree(queue):
    '''
    print the optimal tree (with lowest dissimilarity score) for all budges (all possible model sizes)
    :param queue: queue must have already been sorted - queue.sort(key=lambda x: (x[1], x[0]))
    '''
    # queue must be sorted
    print("Finding the optimal Tree...")

    dic = {}
    for q in queue:
        if q[1] not in dic: # because queue is sorted, the first tree of a new model_size is the optimal one
            dic[q[1]] = q[2]

    for key, value in dic.items():
        print(key, value)


###############################

# for debug, we pre-train all single models in advance
# so that we do not need train them every time
# for i in range(10):
#     model_train(train=True, chosenType=i)

# # for debug, we calculate RSM once and save it
# # so that we do not need to recompute every time
# rsm = RSM_Calc(50)
# np.save('rsm.npy', rsm)
# RSM = np.load('rsm.npy')

RSM = np.load('rsm.npy')
start = time.time()
queue = clustering_withBudget_old(RSM, N=8, Budget=6)
end = time.time()
print('Time spent: {} second'.format(end - start))
# queue = clustering(RSM, N=7)
plotQueue(queue)
optimalTree(queue)


