import tensorflow as tf
from alexnet import AlexNet
import matplotlib.pyplot as plt

image_size = 24
channels = 3
keep_prob = 0.5
num_classes = 10
batch_size = 128
class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def test_image(filename, num_class, weights_path='Default'):
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_jpeg(img_string, channels=channels)
    img_resized = tf.image.resize_images(img_decoded, [image_size, image_size])
    img_reshape = tf.reshape(img_resized, shape=[1, image_size, image_size, channels])

    model = AlexNet(img_reshape, keep_prob, num_classes, batch_size)
    score = tf.nn.softmax(model.fc7)
    max = tf.argmax(score, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./tmp/checkpoint/model_epoch10.ckpt")
        print(max)
        print(score.eval())
        print(max.eval())
        prob = sess.run(max)[0]
        print(prob)
        #print(score.eval()[prob])
        print(class_name[prob])
        #plt.imshow(img_decoded.eval())
        #plt.title("Class: " + class_name[prob])
        #plt.show()

test_image("./test/00002.jpg", num_class=num_classes)