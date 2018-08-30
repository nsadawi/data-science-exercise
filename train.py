import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#load original data
#Please update the path to the files
train = pd.read_csv('/Users/nsadawi/mygit/data-science-exercise/usage_train.csv')

## Pivoting the data
new_train = train.pivot(index='datetime', columns='id', values='usage')

#compute mean or median usage
# at this step the new_train DF can be saved and loaded in predict.py
# saved  only with the mean_usage column to save space and time
new_train['mean_usage'] = new_train.mean(axis=1)
#new_train['median_usage'] = new_train.median(axis=1)


#scale the training data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(new_train['mean_usage'].values.reshape(-1, 1))


## Batch Function
def next_batch(training_data, batch_size, steps):
    # A random starting point for each batch
    rand_start = np.random.randint(0,len(training_data)-steps)

    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)

    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)

### Declare required constants
# Only one feature (the time series)
num_inputs = 1
# Num of steps in each batch (can be changed)
num_time_steps = 48
# 100 neuron layer (can be changed)
num_neurons = 100
# Just one output (the predicted time series)
num_outputs = 1

# learning rate  (can be changed)
learning_rate = 0.03
# how many iterations to go through (training steps) ..  (can be changed)
num_train_iterations = 501
# The size of the batch of data
batch_size = 1

# reset graph to start afresh
tf.reset_default_graph()

# Create Placeholders for X (i.e. the input) and y (the output)
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs],  name = "X")
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs], name = "y")

# Setup one output using OutputProjectionWrapper and prepare the RNN
#Observe that we can use (just examples):
#- BasicRNNCells
#- BasicLSTMCells
#- MultiRNNCell
#- GRUCell
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)


# Now pass in the cells variable into tf.nn.dynamic_rnn, along with the first placeholder (X)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


### Loss Function and Optimizer
## Create a Mean Squared Error Loss Function and use it to minimize an AdamOptimizer
loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

# Initialize the global variables
init = tf.global_variables_initializer()


# Create an instance of tf.train.Saver() so we can save our model
saver = tf.train.Saver()


### The Actual Session

# If we are using a GPU
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    sess.run(init)
    # Here we loop through the training data, get random batches and train the RNN
    for iteration in range(num_train_iterations):

        X_batch, y_batch = next_batch(train_scaled,batch_size,num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        # Print out the training MSE every 100 iterations
        if iteration % 100 == 0:

            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    # Save Model for later
    saver.save(sess, "models/energy_consumption_rnn_model.ckpt")

