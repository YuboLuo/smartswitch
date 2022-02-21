import tensorflow as tf
print('tensorflow version:',tf.__version__)

from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras import backend as F
from tensorflow import keras
from scipy import stats
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import itertools


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
    y_test = [ 1 if c == chosenType else 0 for c in y_test]
    y_train = [1 if c == chosenType else 0 for c in y_train]

    ### expand dimension
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    y_train = tf.expand_dims(y_train, -1)
    y_test = tf.expand_dims(y_test, -1)


    return x_train, x_test, y_train, y_test


def model_train(train = True, chosenType = 0):
    '''
    Retrain a new model or load a pretrained model
    :param train: True means to retrain a model; False means to load a pretrained model
    :param chosenType: the class type you use as 1
    :return: return the keras model
    '''

    if not train:  ### load model data from pretrained model
        model = keras.models.load_model('pretrained/mnist_{}'.format(chosenType))

    else: ### else build and train a new model, and save it

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


        ### compile, train and save model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
        model.save('pretrained/mnist_{}'.format(chosenType))

    return model


def RDM_Calc(K, chosenType = 0, train = False, getFuncNumber = False):

    '''
    calculate the Representation Dissimilarity Matrix (RDM) for each possible branch out point
    this function is for one task, you need to run this function repeatedly for all tasks
    :param K: the number of images to use for calculation, better to use 50 or more
    :param chosenType: the class type you use as 1 for this task
    :param getFuncNumber: if this param is True, we just return the number of func, a.k.a the number of branch out points
    :return: RDM for this task
    '''

    model = model_train(chosenType = chosenType, train = train)
    x_train, x_test, _, _ = dataload_mnist(chosenType)
    images = x_test[:K]

    ### get the name of each layers for debug
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
    T = 10 # number of tasks, mnist dataset has 10 classes and we divide it into 10 individual tasks
    D = RDM_Calc(1, getFuncNumber = True) # number of division points where you possibly branch out
    RSM = np.zeros((D, T, T))

    # calculate RDM for each task
    RDM = [RDM_Calc(K, chosenType = t) for t in range(T)]

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
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller


# we save and reload RSM info
# rsm = RSM_Calc(50)
# np.save('rsm.npy', rsm)
# RSM = np.load('rsm.npy')



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
            tobeAveraged += 0 # the dissimilarity between a task and itself is 0
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


def clustering(RSM):
    '''

    '''
    tasks = list(range(5))
    queue = [] # to save all possible trees, and their score and model size

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
            to see understand itertools.product(*[ ]) works
            '''

            # strip the out layer brackets
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
                         + len(tasks) * sizeByBlock[3] # total model size this tree requires
            tree = [clusters0, '--->' ,clusters1]  # decomposition details

            queue.append([score, model_size, tree])

        # print('\n\n')

    if len(tasks) <= 9:  # if more than 10, queue has 16733779 elements, too slow to be sorted
        queue.sort(key = lambda x:(x[1], x[0]))

    return queue

def plotQueue(queue):
    dic = defaultdict(list)
    for q in queue:
        dic[q[1]].append(q[0])

    for idx, key in enumerate(dic.keys()):
        plt.boxplot(dic[key], positions=[idx], showfliers=False)  # do not plot outliers

    plt.show()


RSM = np.load('rsm.npy')
queue = clustering(RSM)
plotQueue(queue)









##################################################################################################################
########################      the below is what I coded when developing the above functions       ################
##################################################################################################################


# x_train, x_test, y_train, y_test = dataload_mnist(9)
#
# model = models.Sequential()
#
# model.add(layers.Conv2D(8, (7, 7), strides = (2,2), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.Conv2D(8, (2, 2), strides = (2,2), activation='relu', input_shape=(11, 11, 8)))
# model.add(layers.MaxPooling2D((2, 2)))
#
# model.add(layers.Flatten())
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(2))
# # model.summary()
#
# layersName = {idx:[layer.name, layer.output.shape] for idx, layer in enumerate(model.layers)}
#
# ### setup functions to read outputs of intermediate layers
# func0 = F.function([model.layers[0].input], [model.layers[0].output])
# func1 = F.function([model.layers[0].input], [model.layers[2].output])
# func2 = F.function([model.layers[0].input], [model.layers[4].output])
# func3 = F.function([model.layers[0].input], [model.layers[5].output])
# funcList = [func0, func1, func2, func3]
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
#
# model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
#
#
# images = x_test[:100]
#
# K = len(images) # K images were used
# outs = []
# for func in funcList:
#     out = func(images)[0]
#     outs.append(out.reshape(out.shape[0], int(out.size / out.shape[0])))
#
# RDM = np.zeros((len(outs), K, K))
# for out in outs:
#     RDM_f = np.zeros((K, K))  # Representation Dissimilarity Matrix (RDM) for each func (each branch out point)
#     for i in range(K):
#         for j in range(i, K):
#             RDM_f[i][j] = 1 - stats.pearsonr(out[i], out[j])[0]
#             RDM_f[j][i] = RDM_f[i][j]



# res = stats.spearmanr(p1,p2)
# res = res.correlation