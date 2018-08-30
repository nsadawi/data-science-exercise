import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#load original data
#Please update the path to the train and test files
test = pd.read_csv('usage_test.csv')
## Pivoting the test data
new_test = test.pivot(index='datetime', columns='id')# test has no values for household columns


#Please update the path to the files
train = pd.read_csv('usage_train.csv')
## Pivoting the data
new_train = train.pivot(index='datetime', columns='id', values='usage')

#compute mean or median usage
new_train['mean_usage'] = new_train.mean(axis=1)
#new_train['median_usage'] = new_train.median(axis=1)

#scale the training data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(new_train['mean_usage'].values.reshape(-1, 1))

## Predicting Future Usage (Test Data)
## Remember I am using the average consumption to build the model so all
## households will have the same predicted usage. I can easily create models
## and make predictions for individual households but the purpose here is
## just to demonstrate how I can deal with Time Series Data! **

num_points_to_predict = new_test.shape[0]

saver = tf.train.Saver()

# Num of steps in each batch (can be changed)
num_time_steps = 48



with tf.Session() as sess:

    # Use the saver instance to load the saved rnn time series model
    saver.restore(sess,'models/energy_consumption_rnn_model.ckpt')
    #saver.restore(sess, "./energy_consumption_rnn_model.ext")

    # Create a numpy array for the generative seed from the last
    # num_points_to_predict points of the training set data.
    train_seed = list(train_scaled[-num_points_to_predict:])

    ## Now create a for loop to make the predictions
    for iteration in range(num_points_to_predict):
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        train_seed.append(y_pred[0, -1, 0])

# rescale the predicted data
results = scaler.inverse_transform(np.array(train_seed[num_points_to_predict:]).reshape(num_points_to_predict,1))
# create a pandas DF
dataset = pd.DataFrame({'predicted_usage':results.flatten()})
# save it to a csv file
dataset.to_csv('usage_test_predictions.csv', index=False)