"""
    this will involve convolutional layers
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# input output variable
x = tf.placeholder(tf.float32, [None, 784], 'input')
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # for dropout


def conv_2d(x, W, strides=(1, 1, 1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)


def max_pool_2x2(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME'):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def cnn_model(x, weights, biases):
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv_2d(x_reshaped, weights['w_conv1']) + biases['b_conv1'])
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv_2d(h_pool1, weights['w_conv2']) + biases['b_conv2'])
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 10])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['w_dense']) + biases['b_dense'])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_output = tf.add(tf.matmul(h_fc1_drop, weights['w_final']), biases['b_final'], name='y_out')
    return y_output


# create dict of weights/biases for easy reading
weights = {
    'w_conv1': weight_variable([5, 5, 1, 5]),
    'w_conv2': weight_variable([5, 5, 5, 10]),
    'w_dense': weight_variable([7 * 7 * 10, 30]),
    'w_final': weight_variable([30, 10])
}
biases = {
    'b_conv1': bias_variable([5]),
    'b_conv2': bias_variable([10]),
    'b_dense': bias_variable([30]),
    'b_final': bias_variable([10]),
}


# train/test cycle
y_final = cnn_model(x, weights, biases)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_final))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_final, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images,
                                                        y: mnist.test.labels, keep_prob: 1.0}))

    saver = tf.train.Saver()
    saver.save(sess, 'mnist_cnn_model')

    # test 1 data
    x_reshaped = mnist.test.images[0:1]
    print(np.argmax(y_final.eval(feed_dict={x: x_reshaped, keep_prob: 1.0}), axis=1))

