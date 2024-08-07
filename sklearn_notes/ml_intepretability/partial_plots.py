"""
Partial Dependence Plots are calculated after a model has been fit.

 They show how the target variable changes as a function of one or two features, while averaging out the effects of all other features.
 - Particularly useful for identifying non-linear relationships and interactions between features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

data = pd.read_csv('./datasets/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
r = graphviz.Source(tree_graph, format="png")

# Leaves with children show their splitting crieterion
# Pair of values at the bottom show the count of True values and False values for the target variable
disp1 = PartialDependenceDisplay.from_estimator(tree_model, val_X, ['Goal Scored'])
plt.title("Y-axis shows change in prediction from what it would be predicted at leftmost value")
# plt.savefig("partial_dependence_plot.png")
plt.show()

# Y-axis shows the change in prediction from what it would be predicted at the baseline or leftmost value.
# Scoring A goal substantially increases your chances of winning.
# Extra goals beyond appear to have little impact.

# Another Feature
feature_to_plot = "Distance Covered (Kms)"
disp2 = PartialDependenceDisplay.from_estimator(tree_model, val_X, [feature_to_plot])
plt.title("Partial Dependence for Distance Covered (Kms)")
# plt.savefig("partial_dependence_plot_distance.png")
plt.show()

# The partial dependence of a variable can have different implications for different types of models e.g
# If players run over 100 km over the course of a game you are more likely to win, but running very long distances
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
disp3 = PartialDependenceDisplay.from_estimator(rf_model, val_X, [feature_to_plot])
plt.title("Partial Dependence for Distance Covered (Kms) for Random Forest Model")
# plt.savefig("partial_dependence_plot_rf.png")
plt.show()

# ! 2D Partial Dependenc Plots !
fig, ax = plt.subplots(figsize=(8, 6))
f_names = [('Goal Scored', 'Distance Covered (Kms)')]
# Similar to previous PDP plot except we use tuple of features instead of single feature
disp4 = PartialDependenceDisplay.from_estimator(tree_model, val_X, f_names, ax=ax)
plt.title("2D Partial Dependence for Tree Model of Goal Scored and Distance Covered (Kms)")
plt.savefig("partial_dependence_plot_2d.png")
plt.show()

# we see the highest predictions when team scores at least 1 goal + they run a total distance close to 100 km.
# If they score 0 goals, distance covered does not matter.
