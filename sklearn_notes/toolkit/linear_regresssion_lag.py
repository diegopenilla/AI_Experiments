"""

LAG PLOTS

Lag plots are primarily used to identify the following characteristics in time series data:

- Auto-correlation:
    They help in detecting auto-correlation, which is the similarity between observations as a function of the time lag between them.
    Auto-correlation is a key assumption in many time series models, such as ARIMA (AutoRegressive Integrated Moving Average).

- Pattern Identification:
    Lag plots can reveal patterns such as cyclic behavior, trends, or noise in the data.

- Non-Linearity:
    They can indicate if the relationship between observations is non-linear, suggesting the need for non-linear models.

- Outliers:
    They help in identifying outliers that may not be apparent in the original time series plot.

Model Selection:

Before selecting a time series model, a lag plot can help determine whether the data exhibits auto-correlation.
If strong auto-correlation is present, models like ARIMA or exponential smoothing might be appropriate.

DatetimeIndex
Nature:
    A DatetimeIndex represents specific points in time.
    Each entry corresponds to a precise timestamp, which could be down to the millisecond.
Use Case:
    This is useful when each measurement in the time series is taken at an exact, specific moment.
    Examples include stock prices at every minute, sensor readings at every second, or log entries with precise timestamps.
Granularity:
    The index is very granular and precise, capturing the exact moment each event or measurement occurred.

PERIOD INDEX
Nature:
    A PeriodIndex represents time periods rather than specific points in time.
    Each entry corresponds to an interval, such as a day, month, quarter, or year.
Use Case
    This is useful when measurements represent an accumulation over a period.
    Examples include daily total sales, monthly rainfall, quarterly financial results, or yearly population growth.
Aggregation
    It simplifies the process of aggregating and analyzing data over time periods, making it easier to perform operations like resampling, rolling calculations, or period-based comparisons.
"""

from pathlib import Path
from warnings import simplefilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

simplefilter("ignore")  # ignore warnings

# Set Matplotlib defaults
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

# Load Tunnel Traffic dataset
data_dir = Path("./datasets/ts-course-data")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])

# 1. Create a time series in Pandas by setting the index to a date column.
# 2. We parsed "Day" as a date type by using `parse_dates` when loading the data.
tunnel_pre = tunnel.set_index("Day")
tunnel = tunnel_pre.to_period()
tunnel.head()

# 3. create a time dummy by counting out the length of the time series
df = tunnel.copy()
df['Time'] = np.arange(len(tunnel.index))
df.head()

# 4. fitting a linear regression model -> fit trend line to the data
X = df.loc[:, ['Time']]  # feature
y = df['NumVehicles']  # target

model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)
ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Tunnel Traffic');

# 5. Lag Features
df['Lag_1'] = df['NumVehicles'].shift(1) # drop NA rows

X = df.loc[:, ['Lag_1']] # create training feature set
X.dropna(inplace=True)  # drop missing values in the feature set
y = df.loc[:, 'NumVehicles']  # create the target
y, X = y.align(X, join='inner')  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic')
plt.show()



