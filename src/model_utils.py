import os
import numpy as np
import pandas as pd
import pickle
from typing import Union
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from tbats import TBATS
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.metrics import mean_squared_error

# Local Application imports
from src.models import model_lin, sarima, model_tbats



def blocked_time_series_cv(data_length, n_splits, test_size):
    """
    Generates train/validation indices for blocked cross-validation.

    Parameters:
    - data_length: int, length of your time series data
    - n_splits: int, number of folds
    - test_size: int, size of each validation block

    Yields:
    - train_indices: np.array of training indices
    - val_indices: np.array of validation indices
    """
    step = (data_length - test_size) // n_splits
    for i in range(n_splits):
        train_end = step * i + step
        val_start = train_end
        val_end = val_start + test_size

        if val_end > data_length:
            break # Stop if validation window goes past data end

        train_indices = np.arange(train_end)
        val_indices = np.arange(val_start, val_end)

        yield train_indices, val_indices



def best_model_selection(rmse_list, model_list):
    min_index = rmse_list.index(min(rmse_list))
    best_model = model_list[min_index]
    return best_model



def save_best_model(best_model, model_save_path):
    os.makedirs('models', exist_ok=True)
    if isinstance(best_model, SARIMAXResults):
        # Save only the model specification (SARIMAX class + data passed to initializer)
        # so we ccacn refit it later
        spec = best_model.model # SARIMAX spec object
        with open(model_save_path, 'wb') as f:
            pickle.dump(spec, f)
        print("SARIMA specification saved (not fitted instance).")
    elif isinstance(best_model, LinearRegression):
        # Save LinearRegression spec; no fitting yet
        spec = LinearRegression()
        spec.set_params(**best_model.get_params())
        with open (model_save_path, 'wb') as f:
            pickle.dump(spec, f)
    elif isinstance(best_model, TBATS):
        # Save TBATS spec; not fitting yet)
        spec = TBATS(
            seasonal_periods=best_model.seasonal_periods,
            use_box_cox=best_model.use_box_cox,
            use_trend=best_model.use_trend
        )
        with open(model_save_path, 'wb') as f:
            pickle.dump(spec, f)
    else:
        print("Best model does not have save conditions written into this script, nothing saved")


def lag_adjustments(forecast_length, X):
    # We can't pass our linear regression models lag features for times when the lag value would not occur until after the 
    # the model is supposed to be making its forecast - that would be data leakage.
    # Therefore, we need the below conditionals to drop lag features depending on the forecast length. 
    if forecast_length in ['12_hours', 'day']:
        X = X.drop('lag_1', axis=1)
    elif forecast_length == '3_days':
        X = X.drop(columns=['lag_1', 'lag_24'])
    elif forecast_length == 'week':
        X = X.drop(columns=['lag_1', 'lag_24', 'lag_72'])
    return X



def cross_validation(X_train_full,
                     y_train_full,
                     n_splits, 
                     test_size,
                     model_save_path, 
                     forecast_length):
    # We can't pass our linear regression models lag features for times when the lag value would not occur until after the 
    # the model is supposed to be making its forecast - that would be data leakage.
    # Therefore, we need the below conditionals to drop lag features depending on the forecast length. 
    X_train_full = lag_adjustments(forecast_length, X_train_full)

    rmse_lin = []
    rmse_sarima = []
    rmse_tbats = []

    for train_idx, val_idx in blocked_time_series_cv(len(X_train_full), n_splits, test_size):
        X_train, y_train = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]

        model_lin.fit(X_train, y_train)
        model_sarima = sarima(y_train) # this is a function that initalizes sarima with target data, 
        # not a call to fit
        results_sarima = model_sarima.fit()
        results_tbats = model_tbats.fit(y_train)

        
        preds_lin = model_lin.predict(X_val)
        preds_sarima = results_sarima.forecast(steps=len(val_idx))
        preds_tbats = results_tbats.forecast(steps=len(val_idx))

        # Compute RMSE for each model
        rmse_lin.append(np.sqrt(mean_squared_error(y_val, preds_lin)))
        rmse_sarima.append(np.sqrt(mean_squared_error(y_val, preds_sarima)))
        rmse_tbats.append(np.sqrt(mean_squared_error(y_val, preds_tbats)))

    model_lin_rmse = np.mean(rmse_lin)
    model_sarima_rmse = np.mean(rmse_sarima)
    model_tbats_rmse = np.mean(rmse_tbats)

    # Print average RMSE across all folds
    print(f"Linear Model RMSE: {model_lin_rmse:.2f}")
    print(f"SARIMA Model RMSE: {model_sarima_rmse:.2f}")
    print(f"TBATS Model RMSE: {model_tbats_rmse:.2f}")

    results = pd.DataFrame([{'model_lin_rmse': model_lin_rmse, 
                             'model_sarima_rmse': model_sarima_rmse, 
                             'model_tbats_rmse': model_tbats_rmse}])
    
    os.makedirs('cv_rmse_scores', exist_ok=True)
    results.to_csv(f'cv_rmse_scores/rmse_scores_{forecast_length}', index=False)
    
    rmse_list = [model_lin_rmse, model_sarima_rmse, model_tbats_rmse]
    model_list = [model_lin, model_sarima, model_tbats]

    best_model = best_model_selection(rmse_list, model_list)
    print("Best model object type:", type(best_model).__name__)
    save_best_model(best_model, model_save_path)



def load_model(best_model_path: str) -> Union[
      SARIMAXResults, LinearRegression, TBATS]:
    with open(best_model_path, "rb") as f:
            best_model = pickle.load(f)
    return best_model



def rmse(
    y_true: pd.Series, y_pred: pd.Series) -> float:
    # MSE method
    mse = mean_squared_error(y_true, y_pred)
    # Take square root of MSE
    rmse = np.sqrt(mse)
    # Return RMSE and model name
    return rmse



def test(best_model, X_train, y_train, X_test, y_test, best_model_path, forecast_length, steps):
    if isinstance(best_model, SARIMAXResults):
        best_model = SARIMAX(y_train,
                             order=best_model.specification["order"],
                             seasonal_order=best_model.specification["seasonal_order"],
                             enforce_stationarity=best_model.specification["enforce_stationarity"],
                             enforce_invertibility=best_model.specification["enforce_invertibility"])
        results = best_model.fit()
        preds = results.forecast(steps=steps)
    elif isinstance(best_model, LinearRegression):
        X_train = lag_adjustments(forecast_length, X_train)
        X_test = lag_adjustments(forecast_length, X_test)
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)
    elif isinstance(best_model, TBATS):
        results = best_model.fit(y_train)
        preds = results.forecast(steps=steps)
    rmse = rmse(preds, y_test)
    print(f"Best Model: {type(best_model.__name__)}")
    print(f"Best Model RMSE: {rmse}")
    return type(best_model).__name__, rmse