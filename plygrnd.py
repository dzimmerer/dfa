#!/usr/bin/env python
import numpy as np
import tensorflow as tf

import helper

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#### Init

a_0 = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

middle = 100
w_1 = tf.Variable(tf.truncated_normal([784, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, 10]))

w_2_x = tf.Variable(tf.truncated_normal([10, middle]))

#### Functions

def sigma(x):
	return tf.sigmoid(x)
	# return tf.div(tf.constant(1.0),
	# 			  tf.add(tf.constant(1.0), tf.exp(tf.neg(x))))


def sigmaprime(x):
	return tf.mul(sigma(x), tf.sub(tf.constant(1.0), sigma(x)))


#### Frwd

z_1 = tf.matmul(a_0, w_1)
a_1 = sigma(z_1)
z_2 = tf.matmul(a_1, w_2)
a_2 = sigma(z_2)


#### Error

diff = tf.sub(a_2, y)

#### Backward

d_z_2 = tf.mul(diff, sigmaprime(z_2))
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

d_x_1 = tf.matmul(diff, w_2_x)
d_z_1 = tf.mul(d_x_1, sigmaprime(z_1))
d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)

#### Updates

## New (optimizers):

# opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
opt = tf.train.RMSPropOptimizer(learning_rate=0.01)

grads_and_vars2 = [(d_w_1, w_1), (d_w_2, w_2)]
step = opt.apply_gradients(grads_and_vars2)

## Without:

# cost = tf.mul(diff, diff)
# step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


#### Run network


acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(100000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(step, feed_dict = {a_0: batch_xs,
								y : batch_ys})



	if i % 1000 == 0:

		w_imgs = np.reshape(np.swapaxes(w_1.eval(), 0, 1), (middle, 28, 28))
		helper.show_images_quad(w_imgs, Title="xD")

		res = sess.run(acct_res, feed_dict =
					   {a_0: mnist.test.images[:1000],
						y : mnist.test.labels[:1000]})
		print(res)


print("Done.")
