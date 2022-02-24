'''
 - learn how to use tf (old version 1.x) to save and restore models:
    https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
 - Seulk's PerCom paper's repository:
    https://github.com/weight-separation/WeightSeparation
 - purpose of this script:
    to load the pretrained models of Seulki's PerCom paper, to see what network architecture is used
    this file must use tensorflow version <=1.13.2
    tried to use conda (Anaconda) to install tensorflow = 1.13.2 (python = 3.7) and it worked well
'''



import tensorflow as tf
from keras import backend as F

# last three datasets have only 5 layers
datasets = ['mnist', 'fmnist', 'cifar10', 'gtsrb', 'svhn', 'esc10', 'obs', 'gsc', 'hhar', 'us8k']




name_dataset = datasets[5]

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('../../WeightSeparation/{}/{}.meta'.format(name_dataset,name_dataset))  # the downloaded repository is in WeightSeparation folder
    saver.restore(sess, '../../WeightSeparation/{}/{}'.format(name_dataset,name_dataset))

    # n_var = [] # to store number of parameters for each trainable variables - t_var
    # for i in range(len(tf.trainable_variables())):
    #     t_var = tf.trainable_variables()[i]
    #
    #     shape = t_var.shape
    #     l = len(shape)
    #     if l == 4:  # for CNN layers, they has 4 parameters
    #         n_var.append((int(shape[0]) * int(shape[1]) * int(shape[2]) * int(shape[3])))
    #     elif l == 2: # for FC layers, they have 2 parameters
    #         n_var.append((int(shape[0]) * int(shape[1])))
    #     else: # for bias, they have 1 parameter
    #         n_var[-1] += int(shape[0])
    #     print(t_var, n_var[-1])

    t_vars = tf.trainable_variables()
    n_var = []
    for i in range(int(len(t_vars) / 2)):
        n_var.append(F.count_params(t_vars[i * 2]) + F.count_params(t_vars[i * 2 + 1]))
    # print('***',n_var)

    print('\n{} has {} parameters.'.format(name_dataset, sum(n_var)))
    for n in n_var:
        print(n)


    ### if we divide the entire network into three blocks, calculate ratio of each block's weights
    print('\n{} - Ratio:'.format(name_dataset))
    divide = [[0,1,2,3], [4], [5]]
    for block in divide:
        n_b = 0
        for layer in block:
            n_b += n_var[layer]
        print('{0:.2f}'.format(n_b / sum(n_var)))





# graph = tf.get_default_graph()
# names_op = [op.name for op in graph.get_operations()]
# names_tensor = [n.name for n in graph.as_graph_def().node]


