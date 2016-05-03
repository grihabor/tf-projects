import numpy as np
import matplotlib.pyplot as plt
#import tensor flow after pyplot!
import tensorflow as tf

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3 - 0.03 + 0.06*np.random.rand(100).astype(np.float32)

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		x_plt = np.array([-.1, 1.1], dtype=np.float32)
		y_plt = sess.run(W) * x_plt + sess.run(b)
		plt.plot(x_plt, y_plt)
		print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]

plt.scatter(x_data, y_data)
plt.show()
