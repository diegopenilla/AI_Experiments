"""
XGBoost Notes:

1. First, we use the current ensemble to generate predictions for each observation in the dataset.
    To make a prediction, we add the predictions from all models in the ensemble.
2. These predictions are used to calculate a loss function (like mean squared error).
3. Then, we use the loss function to fit a new model that will be added to the ensemble.
- Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss.
 (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)
Finally, we add the new model to ensemble, and ...
4. repeat!

- extreme gradient boosting, performance and speed.
- Parameter tuning:



n_estimators: specifies how many times to go through the modeling cycle described above => Number of models in the ensemble.

- Too low -> underfitting, inaccurate predictions on both training and validation data.
- Too high -> overfitting, accurate predictions on training data but inaccurate predictions on validation data.

Typical values range from 100-1000, though this depends a lot on the learning rate.

-> Early stopping rounds: automatically find the ideal number of n_estimators.
Early stopping rounds -> stop iterating when the validation score stops improving.
Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping.

"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../datasets/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
#
# # Define and train
# my_model = XGBRegressor(n_estimators=500)
# my_model.fit(X_train, y_train,
#              early_stopping_rounds=5,
#              eval_set=[(X_valid, y_valid)],
#              verbose=False)


# multiply predictions from each model by a small number (learning_rate) before adding them in -> the appropriate number of models will be determined automatically.
# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(X_train, y_train,
#              early_stopping_rounds=5,
#              eval_set=[(X_valid, y_valid)],
#              verbose=False)


# On larger datasets, where runtime is a consideration, you can use parallelism to build your models faster.
# On small datasets this won't help.

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

mae = mean_absolute_error(predictions, y_valid)
print("Mean Absolute Error: " + str(mae))