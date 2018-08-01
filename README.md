# USIO Energy Data Science Exercise
You will use historic energy usage data from a number of households to forecast their future
usage.

## Data
`usage_train.csv` contains historic data in a single file.
The columns are:
- `id`: a unique household ID
- `datetime`: datetime of the half-hour interval for which usage is recorded
- `usage`: usage for the given household and half-hour (in kWh)

`usage_test.csv` is similar to `usage_train.csv`, but is missing the `usage` column.
We would like you to predict `usage` for each row (household/half-hour) in `usage_test.csv`.

## Submission
We are interested in how you approach an open-ended machine learning problem.
We encourage plotting and data exploration, as well as readable and re-usable code.

- use `python 3`
- fork this repository
- commit all your code (including Jupyter Notebooks, exploratory work etc.)
- submit a pull request when you are finished
- please make sure the code includes two files:
  * `train.py` should read the training data, perform any steps needed,
    and save a single file containing the trained forecasting model (e.g. `model.ext`)
  * `predict.py` should read the test data (`usage_test.csv`) and the model file (`model.ext`),
    and save predictions in `usage_test_predictions.csv`.
- feel free to organise additional files as you see fit
- please update [`QUESTIONS.md`](QUESTIONS.md) and try to answer questions therein.
- please include a description of external libraries needed to run your code

> Please try to not spend more than half a day on this exercise! Have fun!
