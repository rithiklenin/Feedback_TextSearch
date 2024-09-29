from sklearn.metrics import mean_squared_error
import numpy as np

# Function to evaluate a model's performance
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse
