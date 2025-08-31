# Third-party imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def rmse_calc(
    y_true: pd.Series, y_pred: pd.Series) -> float:
    # MSE method
    mse = mean_squared_error(y_true, y_pred)
    # Take square root of MSE
    rmse = np.sqrt(mse)
    # Return RMSE and model name
    return rmse

def normalized_rmse(y_true, y_pred, method='std'):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    if method == 'std':
        return rmse/np.std(y_true)
    elif method == 'mean':
        return rmse/np.mean(y_true)
    elif method == 'range':
        return rmse/(np.max(y_true) - np.min(y_true))
    else:
        raise ValueError("Method must be either 'std', 'mean', or 'range'")

