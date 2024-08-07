# Calculate and show permutation importance:
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

# Calculate and show partial dependence plot:
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

feature_name = 'Goal Scored'
PartialDependenceDisplay.from_estimator(my_model, val_X, [feature_name])
plt.show()

# Calculate and show Shap Values for One Prediction:
import shap  # package used to calculate Shap values

data_for_prediction = val_X.iloc[0,:]  # use 1 row of data here. Could use multiple rows if desired

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)