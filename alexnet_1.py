from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np
import cifar10
import cifar10_input

batch_size=128
num_batches=30000

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    parameters = []
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_activations(conv2)
        parameters += [kernel, biases]
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)
        parameters += [kernel, biases]

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)
        parameters += [kernel, biases]

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)
        parameters += [kernel, biases]

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)

    with tf.name_scope('fc6') as scope:
        weights = tf.Variable(tf.truncated_normal([6*6*256, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = tf.nn.relu(tf.add(tf.matmul(pool5_flat, weights), biases))
        fc6_drop = tf.nn.dropout(fc6, keep_prob)
        parameters += [weights, biases]
        print_activations(fc6_drop)

    with tf.name_scope('fc7') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        fc7 = tf.nn.relu(tf.add(tf.matmul(fc6_drop, weights), biases))
        fc7_drop = tf.nn.dropout(fc7, keep_prob)
        parameters += [weights, biases]
        print_activations(fc7_drop)

    with tf.name_scope('fc8') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 10], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True, name='biases')
        fc8 = tf.nn.softmax(tf.add(tf.matmul(fc7_drop, weights), biases))
        parameters += [weights, biases]
        print_activations(fc8)

    return fc8, parameters


data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

cifar10.maybe_download_and_extract()

images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])
keep_prob = tf.placeholder(tf.float32)

pred, prameters = inference(image_holder)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label_holder))
train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label_holder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
top_k_op = tf.nn.in_top_k(pred, label_holder, 1)
format_str = ('step %d, loss=%.2f Training accuracy=%.2f (%.1f examples/sec; %.3f sec/batch)')
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for step in range(num_batches):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    sess.run(train_op, feed_dict={image_holder: image_batch, label_holder: label_batch, keep_prob: 0.5})
    duration = time.time() - start_time
    if step%10 == 0:
        examples_per_sec = batch_size/duration
        sec_per_batch = float(duration)
        acc = sess.run(accuracy, feed_dict={image_holder: image_batch, label_holder: label_batch, keep_prob:1.0})
        loss = sess.run(cost, feed_dict={image_holder: image_batch, label_holder: label_batch, keep_prob:1.0})
        print(format_str%(step, loss, acc, examples_per_sec, sec_per_batch))

num_examples = 10000
num_iter = int(math.ceil(num_examples/batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder:label_batch, keep_prob: 1.0})
    true_count += np.sum(predictions)
    step += 1

precision = true_count/total_sample_count
print('precision @ 1 = %.3f'%precision)