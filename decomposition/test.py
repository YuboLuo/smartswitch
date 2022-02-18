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


def model_train(train = True, pretrained = '', chosenType = 0):

    # if not train:  ### load model data from pretrained model
    #     i = 1
    #
    # else: ### build and train a new model

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

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
    return model


def RDM_Calc(K, chosenType = 0):

    model = model_train(chosenType)
    x_train, x_test, _, _ = dataload_mnist(chosenType)
    images = x_test[:K]

    layersName = {idx: [layer.name, layer.output.shape] for idx, layer in enumerate(model.layers)}

    ### setup functions to read outputs of intermediate layers
    func0 = F.function([model.layers[0].input], [model.layers[0].output])
    func1 = F.function([model.layers[0].input], [model.layers[2].output])
    func2 = F.function([model.layers[0].input], [model.layers[4].output])
    func3 = F.function([model.layers[0].input], [model.layers[5].output])
    funcList = [func0, func1, func2, func3]

    K = len(images)  # K images were used
    outs = []
    for func in funcList:
        out = func(images)[0]
        outs.append(out.reshape(out.shape[0], int(out.size / out.shape[0])))

    RDM = np.zeros((len(outs), K, K))
    for idx, out in enumerate(outs):
        # RDM_f = np.zeros((K, K))  # Representation Dissimilarity Matrix (RDM) for each func (each branch out point)
        for i in range(K):
            for j in range(i, K):
                RDM[idx][i][j] = 1 - stats.pearsonr(out[i], out[j])[0]
                RDM[idx][j][i] = RDM[idx][i][j]

    return RDM


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


ans = RDM_Calc(100)
