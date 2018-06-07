import tensorflow as tf
import numpy as np

class AlexNet(object):
    def __init__(self, x, keep_prob, num_classes, batch_size):#skip_layer,weights_path='DEFAULT'
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.BATCH_SIZE = batch_size
        #self.SKIP_LAYER = skip_layer

        #if weights_path == 'DEFAULT':
        #    self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        #else:
        #    self.WEIGHTS_PATH = weights_path

        self.create()

    def print_activations(self, t):
        print(t.op.name, ' ', t.get_shape().as_list())

    def create(self):
        parameters = []
        with tf.name_scope('conv1') as scope:
            # [11, 11, 3, 96]
            kernel = tf.Variable(tf.truncated_normal([5, 5, 3, 32], dtype=tf.float32, stddev=1e-1), name='weights')
            #[1, 4, 4, 1]
            conv = tf.nn.conv2d(self.X, kernel, [1, 2, 2, 1], padding='SAME')
            #shape=[96]
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            self.print_activations(conv1)
            parameters += [kernel, biases]
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        self.print_activations(pool1)

        with tf.name_scope('conv2') as scope:
            #[5, 5, 96, 256]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 48], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            #shape=[256]
            biases = tf.Variable(tf.constant(1.0, shape=[48], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            self.print_activations(conv2)
            parameters += [kernel, biases]
        #pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        #self.print_activations(pool2)

        with tf.name_scope('conv3') as scope:
            #[3, 3, 256, 384]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 48, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
            #shape=[64]
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope)
            self.print_activations(conv3)
            parameters += [kernel, biases]

        with tf.name_scope('conv4') as scope:
            #[3, 3, 64, 64]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            #shape=[64]
            biases = tf.Variable(tf.constant(1.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope)
            self.print_activations(conv4)
            parameters += [kernel, biases]
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool4')
        self.print_activations(pool4)

        #with tf.name_scope('conv5') as scope:
        #    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        #    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        #    biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        #    bias = tf.nn.bias_add(conv, biases)
        #    conv5 = tf.nn.relu(bias, name=scope)
        #    self.print_activations(conv5)
        #    parameters += [kernel, biases]

        #pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        #self.print_activations(pool5)

        with tf.name_scope('fc5') as scope:
            shape = pool4.get_shape().as_list()
            dim = 1
            for d in range(len(shape) - 1):
                dim *= shape[d + 1]
            pool5_flat = tf.reshape(pool4, [-1, dim])
            #n_in = pool5_flat.get_shape()[-1].value
            #pool5_flat = tf.reshape(pool4, [self.BATCH_SIZE, -1])
            dim = pool5_flat.get_shape()[1].value
            weights = tf.Variable(tf.truncated_normal([dim, 256], dtype=tf.float32, stddev=1e-1),
                                  name='weights')
            biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            #pool5_flat = tf.reshape(pool4, [-1, 2 * 2* 64])
            fc5 = tf.nn.relu(tf.add(tf.matmul(pool5_flat, weights), biases))
            fc5_drop = tf.nn.dropout(fc5, self.KEEP_PROB)
            parameters += [weights, biases]
            self.print_activations(fc5_drop)

        with tf.name_scope('fc6') as scope:
            weights = tf.Variable(tf.truncated_normal([256, 256], dtype=tf.float32, stddev=1e-1),
                                  name='weights')
            biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            fc6 = tf.nn.relu(tf.add(tf.matmul(fc5_drop, weights), biases))
            fc6_drop = tf.nn.dropout(fc6, self.KEEP_PROB)
            parameters += [weights, biases]
            self.print_activations(fc6_drop)

        with tf.name_scope('fc7') as scope:
            weights = tf.Variable(tf.truncated_normal([256, self.NUM_CLASSES], dtype=tf.float32, stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[self.NUM_CLASSES], dtype=tf.float32), trainable=True, name='biases')
            self.fc7 = tf.add(tf.matmul(fc6_drop, weights), biases)
            parameters += [weights, biases]
            self.print_activations(self.fc7)

        #with tf.name_scope('fc6') as scope:
        #    weights = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], dtype=tf.float32, stddev=1e-1),
        #                          name='weights')
        #    biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        #    pool5_flat = tf.reshape(pool5, [-1, 6 * 6 * 256])
        #    fc6 = tf.nn.relu(tf.add(tf.matmul(pool5_flat, weights), biases))
        #    fc6_drop = tf.nn.dropout(fc6, self.KEEP_PROB)
        #    parameters += [weights, biases]
        #    self.print_activations(fc6_drop)

        #with tf.name_scope('fc7') as scope:
        #    weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
        #    biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        #    fc7 = tf.nn.relu(tf.add(tf.matmul(fc6_drop, weights), biases))
        #    fc7_drop = tf.nn.dropout(fc7, self.KEEP_PROB)
        #    parameters += [weights, biases]
        #    self.print_activations(fc7_drop)

        #with tf.name_scope('fc8') as scope:
        #    weights = tf.Variable(tf.truncated_normal([4096, self.NUM_CLASSES], dtype=tf.float32, stddev=1e-1), name='weights')
        #    biases = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True, name='biases')
        #    fc8 = tf.nn.softmax(tf.add(tf.matmul(fc7_drop, weights), biases))
        #    parameters += [weights, biases]
        #    self.print_activations(fc8)

    #def load_initial_weights(self, session):
    #    weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').items()
    #    for op_name in weights_dict:
    #        if op_name not in self.SKIP_LAYER:
    #            with tf.variable_scope(op_name, reuse=True):
    #                for data in weights_dict[op_name]:
    #                    if len(data.shape) == 1:
    #                        var = tf.get_variable('biases', trainable=False)
    #                        session.run(var.assign(data))
    #                    else:
    #                        var = tf.get_variable('weights', trainable=False)
    #                        session.run(var.assign(data))


