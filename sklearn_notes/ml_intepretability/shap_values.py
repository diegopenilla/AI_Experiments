"""
Shows features each contributing to push the model output from base value
Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.
"""
import xgboost
import shap

# train an XGBoost model
X, y = shap.datasets.california()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[0], show=True)
