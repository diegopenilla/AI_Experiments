"""

target = weight * time + bias

Lag features let you model serial dependence.
A time series has serial dependence when an observation can be predicted from previous observations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "./datasets/ts-course-data/book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

# create time dummy variable -> counts the number of steps since the first observation
df['Time'] = np.arange(len(df.index))
df.head()

plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

# lag features let us fit curves to LAG PLOTS where each observation in a series is plotted against the previous observation
df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'), marker='x')
ax.set_title('Time Plot of Hardcover Sales')
plt.show()

# 1 step lag feature
# linear regression with lag results in target = weight * lag + bias


