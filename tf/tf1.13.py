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
import tensorflow.contrib.slim as slim



# last three datasets have only 5 layers
datasets = ['mnist', 'fmnist', 'cifar10', 'gtsrb', 'svhn', 'esc10', 'obs', 'gsc', 'hhar', 'us8k']




name_dataset = datasets[0]

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('../../WeightSeparation/{}/{}.meta'.format(name_dataset,name_dataset))  # the downloaded repository is in WeightSeparation folder
    saver.restore(sess, '../../WeightSeparation/{}/{}'.format(name_dataset,name_dataset))

    t_vars = tf.trainable_variables()  # obtain trainable variables
    slim.model_analyzer.analyze_vars(t_vars, print_info=True)  # print model summary

    n_var = []
    for i in range(int(len(t_vars) / 2)):
        '''
        there are two ways of counting parameters of trainable variables: 
            (method 1) use F.count_params, you have to first 'from keras import backend as F'
            (method 2) use built-in function: var.get_shape().num_elements()
        '''
        # n_var.append(F.count_params(t_vars[i * 2]) + F.count_params(t_vars[i * 2 + 1])) # method1
        n_var.append(t_vars[i * 2].get_shape().num_elements() + t_vars[i * 2 + 1].get_shape().num_elements()) # method2

    # print('***',n_var)

    print('\nModel {} has {} parameters and {} layers.'.format(name_dataset, sum(n_var), len(n_var)))
    for i,n in enumerate(n_var):
        print('layer {}: {}'.format(i, n))


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


