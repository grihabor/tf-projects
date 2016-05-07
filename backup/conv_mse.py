import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os 
from skimage import transform

#unused
var_list = []

batch_size = 1
out_size = 256*256

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
    i = np.load(filename).astype(np.float32)
    filename = output_dir + 'batch{}.npy'.format(index)
    o = np.load(filename).astype(np.int32)

    indices = np.random.choice(i.shape[0], n, replace=False)
    batch = [i[indices], o[indices]]
    '''
    print(batch[0].shape)
    print(batch[1].shape)
    '''
    batch[0] = batch[0].reshape((batch[0].shape[0], -1))
    batch[1] = batch[1].reshape((batch[1].shape[0], -1))
    '''
    plt.subplot(121)
    plt.imshow(i[0], cmap='gray', interpolation='None')
    plt.subplot(122)
    plt.imshow(o[0], cmap='gray', interpolation='None')
    plt.show()
    '''
    return batch

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial)
    var_list.append(var)
    return var

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    var = tf.Variable(initial)
    var_list.append(var)
    return var

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 256*256])
y_ = tf.placeholder(tf.int32, shape=[None, out_size])

x_image = tf.reshape(x, [-1, 256, 256, 1])

#First Convolutional Layer
W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#[-1, 128, 128, 64]

#Second Convolutional Layer
W_conv2 = weight_variable([3, 3, 64, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#[-1, 64, 64, 128]

#Third Convolutional Layer
W_conv3 = weight_variable([3, 3, 128, 256])
b_conv3 = bias_variable([256])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
#[-1, 32, 32, 256]

#Fourth Convolutional Layer
W_conv4 = weight_variable([3, 3, 256, 512])
b_conv4 = bias_variable([512])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)
#[-1, 16, 16, 512]

#Fifth Convolutional Layer
W_conv5 = weight_variable([7, 7, 512, 4096])
b_conv5 = bias_variable([4096])

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
#[-1, 16, 16, 4096]

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_drop1 = tf.nn.dropout(h_conv5, keep_prob)

#Sixth Convolutional Layer
W_conv6 = weight_variable([1, 1, 4096, 4096])
b_conv6 = bias_variable([4096])

h_conv6 = tf.nn.relu(conv2d(h_drop1, W_conv6) + b_conv6)
#[-1, 16, 16, 4096]

#Dropout
h_drop2 = tf.nn.dropout(h_conv6, keep_prob)

#Seventh Convolutional Layer
W_conv7 = weight_variable([1, 1, 4096, 64])
b_conv7 = bias_variable([64])

h_conv7 = tf.nn.relu(conv2d(h_drop2, W_conv7) + b_conv7)
#[-1, 16, 16, 32]

#Deconvolutional Layer
W_deconv = weight_variable([64, 64, 3, 64])
b_deconv = bias_variable([64])

y_conv = tf.nn.conv2d_transpose(
    h_conv7 + b_deconv, 
    W_deconv,
    [batch_size, 256, 256, 3],
    #y_.get_shape(),
    [1,16,16,1], 
    padding='SAME')

#print(transpose)
#y_conv = tf.Print(y_conv, [y_conv], "y_conv: ")
'''
num_classes == 3
'''
y_conv = tf.reshape(y_conv, [-1, 3])

y_in = tf.to_int64(tf.reshape(y_, [-1]))

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_in)
cross_entropy = tf.reduce_mean(cross_entropy)
cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cross_entropy: ")


saver = tf.train.Saver()

y_prob = tf.nn.softmax(y_conv)
y_fin = tf.one_hot(y_in, 3, 1., 0.)
print('y_prob shape: ', y_prob.get_shape())
print('y_fin  shape: ', y_fin.get_shape())



squared_error = tf.reduce_sum(tf.nn.l2_loss(y_prob - y_fin))
accuracy = tf.reduce_mean(tf.div(squared_error, tf.reduce_sum(y_fin)))

train_step = tf.train.AdamOptimizer(1e-8).minimize(squared_error)

SHOW_PLOTS = False

sess.run(tf.initialize_all_variables())

#saver.restore(sess, 'my-model-50')

for i in range(20000):
    batch = next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    #print(squared_error.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.}))

    if i%100 == 0:

        inp = y_fin.eval({y_: batch[1]})
        inp = np.reshape(inp, (batch_size, 256, 256, 3))
        out = y_prob.eval(feed_dict={x:batch[0], keep_prob: 1.})
        out = np.reshape(out, (batch_size, 256, 256, 3))
        
        #print(out)
        '''
        t = (h_conv7.eval(feed_dict={x:batch[0], keep_prob: 1.}))
        for tt in t:
            for ttt in tt:
                for tttt in ttt:
                    print(tttt)
        
        '''
        if SHOW_PLOTS:
            for j in range(3):
                plt.subplot(2, 3, j+1)
                plt.imshow(inp[0, :, :, j], cmap='gray', interpolation='None')
                plt.subplot(2, 3, j+4)
                plt.imshow(out[0, :, :, j], cmap='gray', interpolation='None')
            plt.show()
        
        
        print(saver.save(sess, 'my-model', global_step=i))

        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.})
        print("step {}, training accuracy {}".format(i, train_accuracy))
        print("squared error {}".format(squared_error.eval({x:batch[0], y_:batch[1], keep_prob: 1.})))
        print("sum {}".format(tf.reduce_sum(y_fin).eval({ y_:batch[1]})))

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.}))



