from datetime import datetime
import os
import math
import time
import tensorflow as tf
import numpy as np
import cifar10
import cifar10_input
from alexnet import AlexNet

image_size = 24
channels = 3
batch_size = 100
num_epoches = 10
num_classes = 10
train_batches_per_epoch = 5000
test_batches_per_epoch = 1000
learning_rate = 1e-3
dropout_rate = 0.5
display_step = 10
filewriter_path = "./tmp/tensorboard"
checkpoint_path = "./tmp/checkpoint"
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

cifar10.maybe_download_and_extract()

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, channels])
label_holder = tf.placeholder(tf.int32, [batch_size])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(image_holder, keep_prob, num_classes, batch_size)

prediction = model.fc7

#label_holder = tf.cast(label_holder, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=label_holder, name='cross_entropy_per_example')
loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label_holder, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
top_k_op = tf.nn.in_top_k(prediction, label_holder, 1)
format_str = ('epoch %d: step %d, loss=%.3f')

tf.summary.scalar('loss', loss)
#tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(filewriter_path, sess.graph)
saver = tf.train.Saver()

tf.global_variables_initializer().run()
tf.train.start_queue_runners()

print("{} Start training...".format(datetime.now()))
print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

for epoch in range(num_epoches):
    print("{} Epoch number: {} start".format(datetime.now(), epoch+1))
    train_acc = 0
    train_count = 0
    for step in range(train_batches_per_epoch):
        image_batch, label_batch = sess.run([images_train, labels_train])
        sess.run(train_op, feed_dict={image_holder: image_batch, label_holder: label_batch, keep_prob: dropout_rate})
        acc = sess.run(top_k_op, feed_dict={image_holder: image_batch, label_holder: label_batch, keep_prob: 1.0})
        train_acc += np.sum(acc)
        train_count += 1
        if step % display_step == 0:
            losses = sess.run(loss, feed_dict={image_holder: image_batch, label_holder: label_batch, keep_prob: 1.0})
            s = sess.run(merged_summary, feed_dict={image_holder: image_batch, label_holder: label_batch, keep_prob: 1.0})
            writer.add_summary(s, epoch * train_batches_per_epoch + step)
            print(format_str % (epoch+1, step, losses))
    train_acc /= train_count
    print("{} Training Accuracy = {:.4f}".format(datetime.now(), train_acc))

    print("{} Start validation".format(datetime.now()))
    test_acc = 0.
    test_count = 0
    for _ in range(test_batches_per_epoch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        acc = sess.run(top_k_op, feed_dict={image_holder: image_batch, label_holder: label_batch, keep_prob: 1.0})
        test_acc += np.sum(acc)
        test_count += 1
    test_acc /= test_count
    print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

    print("{} Saving checkpoint of model...".format(datetime.now()))

    checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch+1) + '.ckpt')
    save_path = saver.save(sess, checkpoint_name)

    print("{} Epoch number: {} end".format(datetime.now(), epoch+1))

writer.close()





