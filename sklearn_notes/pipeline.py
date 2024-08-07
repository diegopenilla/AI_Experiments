"""
Steps:
1. Load the dataset and separate it into features (X) and target variable (y).
2. Split the data into training and validation sets.
3. Identify and select categorical columns with low cardinality and numerical columns.
4. Preprocess the data:
    - Handle missing values in numerical columns using a constant strategy.
    - Impute missing values in categorical columns using the most frequent strategy and apply one-hot encoding.
5. Combine preprocessing steps for numerical and categorical data using ColumnTransformer.
6. Create and configure a RandomForestRegressor model.
7. Construct a machine learning pipeline that bundles the preprocessing and modeling steps.
8. Train the model on the training data.
9. Validate the model on the validation data and evaluate its performance using Mean Absolute Error (MAE).

Modules:
- pandas: For data manipulation and analysis.
- sklearn.model_selection: For splitting the data into training and validation sets.
- sklearn.compose: For constructing a preprocessing pipeline that handles both numerical and categorical data.
- sklearn.pipeline: For creating a machine learning pipeline.
- sklearn.impute: For handling missing values.
- sklearn.preprocessing: For applying one-hot encoding to categorical features.
- sklearn.ensemble: For using a RandomForestRegressor model.
- sklearn.metrics: For evaluating the model's performance using Mean Absolute Error (MAE).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
data = pd.read_csv('../datasets/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [
    cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
]

# Select numerical columns
numerical_cols = [
    cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']
]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_train.head()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    # this handles missing values in datasets: most_frequent -> or mean/median
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # allows model to handle unseen categories gracefully avoiding errors
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, random_state=0)

# 3. Model
from sklearn.metrics import mean_absolute_error
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# 4. Train
my_pipeline.fit(X_train, y_train)

# 5. Validate
preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)


# cross validation
from sklearn.model_selection import cross_val_score
# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(
    my_pipeline, X, y,
    cv=5,
    scoring='neg_mean_absolute_error'
)

print("MAE scores:\n", scores)
print("Average MAE score (across experiments):")
print(scores.mean())