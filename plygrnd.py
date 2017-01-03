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
top = 30
w_1 = tf.Variable(tf.truncated_normal([784, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, top]))
w_3 = tf.Variable(tf.truncated_normal([top, 10]))

w_2_b = tf.Variable(tf.truncated_normal([middle, top]))
w_3_b = tf.Variable(tf.truncated_normal([top, 10]))

#### Functions

def sigma(x):
	return tf.div(tf.constant(1.0),
				  tf.add(tf.constant(1.0), tf.exp(tf.neg(x))))


def sigmaprime(x):
	return tf.mul(sigma(x), tf.sub(tf.constant(1.0), sigma(x)))


#### Frwd

z_1 = tf.matmul(a_0, w_1)
a_1 = sigma(z_1)
z_2 = tf.matmul(a_1, w_2)
a_2 = sigma(z_2)
z_3 = tf.matmul(a_2, w_3)
a_3 = sigma(z_3)

#### Error

diff = tf.sub(a_3, y)

#### Backward

d_z_3 = tf.mul(diff, sigmaprime(z_3))
d_w_3 = tf.matmul(tf.transpose(a_2), d_z_3)

d_a_2 = tf.matmul(d_z_3, tf.transpose(w_3_b))
d_z_2 = tf.mul(d_a_2, sigmaprime(z_2))
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2_b))
d_z_1 = tf.mul(d_a_1, sigmaprime(z_1))
d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)

#### Updates

eta = tf.constant(0.5)
step = [
	tf.assign(w_1, tf.sub(w_1, tf.mul(eta, d_w_1)))
	, tf.assign(w_2, tf.sub(w_2, tf.mul(eta, d_w_2)))
	, tf.assign(w_3, tf.sub(w_3, tf.mul(eta, d_w_3)))
]

### "New:"
#cost = tf.mul(diff, diff)
#step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


#### Run network


acct_mat = tf.equal(tf.argmax(a_3, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(100000):
	batch_xs, batch_ys = mnist.train.next_batch(10)
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
