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


buffer = [None, None]
buffer_count = 0

def next_batch(n):
    global buffer_count
    if buffer_count == 0:
        buffer_count = 50
        index = np.random.randint(0, num_files)

        filename = input_dir + 'batch{}.npy'.format(index)
        buffer[0] = np.load(filename).astype(np.float32)
        filename = output_dir + 'batch{}.npy'.format(index)
        buffer[1] = np.load(filename).astype(np.int32)

    indices = np.random.choice(buffer[0].shape[0], n, replace=False)
    batch = [buffer[0][indices], buffer[1][indices]]
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
    buffer_count -= 1
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

def conv2d(x, W, pad='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=pad)

def max_pool(bottom, ks=2, stride=2):
    return tf.nn.max_pool(bottom, ksize=[1, ks, ks, 1],
                            strides=[1, stride, stride, 1], padding='SAME')

def conv_relu(bottom, nin, nout, ks=3, stride=1, pad='SAME'):
    W = weight_variable([ks, ks, nin, nout])
    b = bias_variable([nout])
    conv = conv2d(bottom, W, pad=pad)
    return conv, tf.nn.relu(conv + b)



sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 256*256])
y_ = tf.placeholder(tf.int32, shape=[None, out_size])

x_image = tf.reshape(x, [-1, 256, 256, 1])


conv1_1, relu1_1 = conv_relu(x_image, 1, 64)
pool1 = max_pool(relu1_1)

conv2_1, relu2_1 = conv_relu(pool1, 64, 128)
pool2 = max_pool(relu2_1)

conv3_1, relu3_1 = conv_relu(pool2, 128, 256)
conv3_2, relu3_2 = conv_relu(relu3_1, 256, 256)
pool3 = max_pool(relu3_1)

conv4_1, relu4_1 = conv_relu(pool3, 256, 512)
pool4 = max_pool(relu4_1)

conv5_1, relu5_1 = conv_relu(pool4, 512, 512)
pool5 = max_pool(relu5_1)

# fully conv
fc6, relu6 = conv_relu(pool5, 512, 4096, ks=7, pad='VALID')

#Dropout
keep_prob = tf.placeholder(tf.float32)
drop6 = tf.nn.dropout(relu6, keep_prob)

fc7, relu7 = conv_relu(drop6, 4096, 4096, ks=1)

#Dropout
drop7 = tf.nn.dropout(relu7, keep_prob)

score_fr, relu8 = conv_relu(drop7, 4096, 32, ks=1)


'''
#deconv
W_deconv = weight_variable([128, 128, 3, 32])
b_deconv = bias_variable([32])
y_conv = tf.nn.conv2d_transpose(
    score_fr - b_deconv, 
    W_deconv,
    [batch_size, 256, 256, 3],
    #y_.get_shape(),
    [1,128,128,1],
    padding='SAME')
'''

W_fc = weight_variable([2*2*32, 32*32*3])
b_fc = bias_variable([32*32*3])

y_last = tf.reshape(tf.matmul(tf.reshape(relu8, [-1, 2*2*32]), W_fc) + b_fc, [-1, 32, 32, 3])          

y_resized = tf.image.resize_nearest_neighbor(y_last, [256, 256])

y_conv = tf.reshape(y_resized, [-1, 3])

y_in = tf.to_int64(tf.reshape(y_, [-1]))
y_fin = tf.one_hot(y_in, 3, 1., 0.)



'''
#fc8, relu8 = conv_relu(y_conv, 3, 3, ks=1)
y_conv = tf.reshape(y_conv, [-1, 3])
'''

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_fin))
cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cross_entropy: ")


squared_error = tf.reduce_sum(tf.squared_difference(y_conv, y_fin))
squared_error = tf.Print(squared_error, [squared_error], "squared_error: ")

accuracy = tf.reduce_mean(tf.div(cross_entropy, tf.reduce_sum(y_fin)))



train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver()

y_prob = tf.nn.softmax(y_conv)
print('y_prob shape: ', y_prob.get_shape())
print('y_fin  shape: ', y_fin.get_shape())

sess.run(tf.initialize_all_variables())

SHOW_PLOTS = True

#saver.restore(sess, 'my-model-9499')

for i in range(20000):
    batch = next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    #print(squared_error.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.}))

    if i%500 == 0:

        inp = y_fin.eval({y_: batch[1]})
        inp = np.reshape(inp, (batch_size, 256, 256, 3))
        out = y_prob.eval(feed_dict={x:batch[0], keep_prob: 1.})
        out = np.reshape(out, (batch_size, 256, 256, 3))
        
        #print(score_fr.eval({x:batch[0], keep_prob: 1.}))
        #print(y_conv.eval({x:batch[0], keep_prob: 1.}))
        print(y_prob.eval({x:batch[0], keep_prob: 1.}))

        if SHOW_PLOTS:
            for j in range(3):
                plt.subplot(2, 3, j+1)
                plt.imshow(inp[0, :, :, j], cmap='gray', interpolation='None')
                plt.subplot(2, 3, j+4)
                plt.imshow(out[0, :, :, j], cmap='gray', interpolation='None')
            plt.show()
        
        
        #print(saver.save(sess, 'my-model', global_step=i))

        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.})
        print("step {}, training accuracy {}".format(i, train_accuracy))
        print("squared error {}".format(squared_error.eval({x:batch[0], y_:batch[1], keep_prob: 1.})))
        print("sum {}".format(tf.reduce_sum(y_fin).eval({ y_:batch[1]})))

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.}))



