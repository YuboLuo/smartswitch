import tensorflow as tf
print('tensorflow version:',tf.__version__)

from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras import backend as F
from tensorflow import keras
from scipy import stats
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


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


def RDM_Calc(K, chosenType = 0, train = False):
    '''
    calculate the Representation Dissimilarity Matrix (RDM) for each possible branch out point
    this function is for one task, you need to run this function repeatedly for all tasks
    :param K: the number of images to use for calculation, better to use 50 or more
    :param chosenType: the class type you use as 1 for this task
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
    func0 = F.function([model.layers[0].input], [model.layers[0].output])
    func1 = F.function([model.layers[0].input], [model.layers[2].output])
    func2 = F.function([model.layers[0].input], [model.layers[4].output])
    func3 = F.function([model.layers[0].input], [model.layers[5].output])
    funcList = [func0, func1, func2, func3]

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
    calculate the task-wise Representation Similarity Matrix (RSM)
    :param K: the number of images to use for RDM calculation
    :return:
    '''

    ### some parameters
    T = 10 # number of tasks, mnist dataset has 10 classes and we divide it into 10 individual tasks
    D = 4 # number of division points where you possibly branch out
    RSM = np.zeros((D, T, T))

    ### calculate RDM for each task
    RDM = [RDM_Calc(K, chosenType = t) for t in range(T)]

    ###
    for d in range(D):
        for i in range(T):
            for j in range(T):

                ### extract RDM of the d_th division point for task_i and task_j
                m1, m2 = RDM[i][d], RDM[j][d]

                ### extract the upper triangle of the matrix and flatten them into a list
                p1 = [elem for ii, row in enumerate(m1) for jj, elem in enumerate(row) if ii < jj]
                p2 = [elem for ii, row in enumerate(m2) for jj, elem in enumerate(row) if ii < jj]

                ### calcualte the Spearman’s correlation coefficient for task_i and task_j at the d_th division point
                RSM[d][i][j] = stats.spearmanr(p1, p2).correlation
    return RSM

rsm = RSM_Calc(50)





# ans = RDM_Calc(50, chosenType = 9, train = True)




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