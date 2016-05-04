import tensorflow as tf
import numpy as np
import os 

data_dir = '../data/'
input_dir = data_dir + 'input/'
output_dir = data_dir + 'output/'

num_files = 0
while 1:
    filename = input_dir + 'batch{}.npy'.format(num_files)
    if not os.path.exists(filename):
        break
    num_files += 1

#print(num_files)

def next_batch(n):
    index = np.random.randint(0, num_files)
    filename = input_dir + 'batch{}.npy'.format(index)
    i = np.load(filename)
    filename = output_dir + 'batch{}.npy'.format(index)
    o = np.load(filename)
    indices = np.random.choice(i.shape[0], n, replace=False)
    batch = [i[indices], o[indices]]
    batch[0] = batch[0].reshape((batch[0].shape[0], batch[0].shape[1]*batch[0].shape[2]))
    batch[1] = batch[1, :, :, 0].reshape((batch[1].shape[0], batch[1].shape[1]*batch[1].shape[2]))
    return batch

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 256*256])
y_ = tf.placeholder(tf.float32, shape=[None, 256*256*2])

x_image = tf.reshape(x, [-1, 256, 256, 1])

#First Convolutional Layer
W_conv1 = weight_variable([8, 8, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second Convolutional Layer
W_conv2 = weight_variable([8, 8, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Third Convolutional Layer
W_conv3 = weight_variable([8, 8, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
print(h_pool3)

#Deconvolutional Layer
W_conv4 = weight_variable([32, 32, 1, 128])
b_conv4 = bias_variable([1])


#todo smth wrong with output shape
output_shape = np.array([50, 32, 32, 1], dtype=np.int32)
print(output_shape)
transpose = tf.nn.conv2d_transpose(
    h_pool3, 
    W_conv4,
    output_shape,
    [1,1,1,1], 
    padding='SAME')

print(transpose)
y_conv = tf.nn.relu(transpose + b_conv4)
y_conv = tf.reshape(y_conv, [-1, 256*256])
print(y_conv)

#training
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
print(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.greater(tf.constant(.1), tf.div(tf.reduce_sum(tf.squared_difference(y_conv, y_), [1]), tf.reduce_sum(y_, [1])))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))



