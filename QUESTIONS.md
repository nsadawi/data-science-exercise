# Questions
Please try to answer the following questions.
Reference your own code/Notebooks where relevant.

## Your solution
1. What machine learning techniques did you use for your solution? Why?

I used recurrent neural networks (RNNs) using Google's tensorflow. They are
suitable for modeling time series data.
2. What is the error of your prediction?

It is not possible to measure the error of the predictions because we do not
have actual values for the test data.
   a. How did you estimate the error?
   Not for the test data. For the training error I minimize MSE
   b. How do the train and test errors compare?
Not applicable for the above reason (we do not
have actual values for the test data)
   c. How will this error change for predicting a further week/month/year into the future?


3. What improvements to your approach would you pursue next? Why?
Try different types of RNNs

Try different values of parameters (e.g. learning rate and iterations)
4. Will your approach work for a new household with little/no half-hourly data?
Yes it should work because I am using the average as I explain in detail in
the notebook.

   How would you approach forecasting for a new household?
Because I use the average, I can predict an average usage using the trained
model!