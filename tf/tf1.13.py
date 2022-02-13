# https://www.google.com/search?q=tensorflow+meta+index+files&oq=tensorflow+meta+index+files&aqs=chrome..69i57j69i64.6113j0j4&sourceid=chrome&ie=UTF-8

import tensorflow as tf

name_dataset = 'us8k'

sess = tf.Session()
saver = tf.train.import_meta_graph('../../WeightSeparation/{}/{}.meta'.format(name_dataset,name_dataset))
saver.restore(sess, '../../WeightSeparation/{}/{}'.format(name_dataset,name_dataset))

print(name_dataset)
n_var = 0
for i in range(len(tf.trainable_variables())):
    t_var = tf.trainable_variables()[i]

    shape = t_var.shape
    l = len(shape)
    if l == 4:
        n_var += (int(shape[0]) * int(shape[1]) * int(shape[2]) + int(shape[3]))
    elif l == 2:
        n_var += (int(shape[0]) * int(shape[1]))
    else:
        n_var += int(shape[0])


    print(t_var)

# graph = tf.get_default_graph()
# names_op = [op.name for op in graph.get_operations()]
# names_tensor = [n.name for n in graph.as_graph_def().node]

