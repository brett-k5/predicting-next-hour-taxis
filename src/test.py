# Standard library imports
import statistics as st
from typing import Union

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from tbats import TBATS

# Local application imports
from src.metrics import normalized_rmse, rmse_calc


def test(best_model: Union[SARIMAXResults, LinearRegression, TBATS], 
         X_train: pd.DataFrame, 
         y_train: pd.Series, 
         X_test: pd.DataFrame, 
         y_test: pd.Series,
         forecast_length: str,
         forecast_length_int: int) -> tuple[str, float, float, float]:
    """
    Performs rolling-origin evaluation of a time series forecasting model.

    The function incrementally expands the training set. It then re-fits and re-tests the model at each step,
    generating multi-step forecasts. It supports SARIMA, Linear Regression, and TBATS models,
    and computes forecast accuracy using RMSE, normalized RMSE, and R² score.

    Predictions are made for the specified forecast horizon (`forecast_length_int`). 
    True and predicted values are accumulated to compute overall performance metrics.

    Args:
        best_model: A fitted forecasting model instance. Supported models include:
            - statsmodels' SARIMAXResults
            - sklearn's LinearRegression
            - tbats.TBATS
        X_train (pd.DataFrame): Initial training features (used only for models that require exogenous variables).
        y_train (pd.Series): Initial training target values.
        X_test (pd.DataFrame): Test features to use in each step of rolling forecast.
        y_test (pd.Series): Test target values to evaluate forecast accuracy.
        forecast_length (str): Forecast horizon label (e.g., '1h', '3h'), used for reporting.
        forecast_length_int (int): Number of future steps to forecast at each iteration.

    Returns:
        tuple:
            model_name (str): Name of the model used for evaluation.
            rmse (float): Root Mean Squared Error over all rolling steps.
            normalized_rmse (float): RMSE divided by standard deviation of actual values.
            r2 (float): R² score measuring forecast accuracy.

    Raises:
        ValueError: If the provided model is not one of the supported types.

    Notes:
        - Each model is re-fitted at the current training window. 
        - For TBATS and SARIMA which are more computationally expensive, this can lead to long run times.
        - The rolling window expands with each step to simulate real-world forecasting conditions.
    """
    
    # Initialize empty predictions lists - to be populated with pandas Series 
    preds_sarima, preds_lin, preds_tbats = [], [], []
    y_test_all_steps = []

    # rolling_steps is the number of data points that will be iteratively added to the training set 
    # Therefore, it will also be the number of times the model re-trains
    rolling_steps = len(X_test) - forecast_length_int + 1
    for step in range(rolling_steps):
        
        # Print step number to help track progress
        print(f"\n\nStep Number: {step}\n\n")

        # Define rolling train and test windows
        X_train = pd.concat([X_train, X_test.iloc[:step]])
        y_train = pd.concat([y_train, y_test.iloc[:step]])
        X_test_step = X_test.iloc[step : step + forecast_length_int]
        y_test_step = y_test.iloc[step : step + forecast_length_int]

        if len(X_test_step) < forecast_length_int:
            break # not enough data left for full forecast
        
        # Add true target values for current loop to y_test_all_steps list
        y_test_all_steps.append(y_test_step)
        
        # Refit best_model on full training set, append empty predictions lists with preds from current loop
        if isinstance(best_model, SARIMAXResults):
            best_model = SARIMAX(y_train, # if best_model is an instance of SARIMAXResults, 
                                order=best_model.specification["order"], # fit SARIMA model to full y_train set with 
                                seasonal_order=best_model.specification["seasonal_order"], # hyperparameters set as the hyperparameters
                                enforce_stationarity=best_model.specification["enforce_stationarity"], # from best_model
                                enforce_invertibility=best_model.specification["enforce_invertibility"])
            results = best_model.fit()
            preds = results.forecast(steps=forecast_length_int)
            preds = pd.Series(preds)
            preds_sarima.append(preds)
        elif isinstance(best_model, LinearRegression):
            best_model.fit(X_train, y_train) # Fit LinearRegression model to full training set
            preds = best_model.predict(X_test_step)
            preds = pd.Series(preds)
            preds_lin.append(preds)
        elif isinstance(best_model, TBATS):
            results = best_model.fit(y_train) # Fit TBATS model to full training set 
            preds = results.forecast(steps=forecast_length_int)
            preds = pd.Series(preds)
            preds_tbats.append(preds)

    # Concatinate pandas series in target and preds lists and calculate and print metrics
    y_test_all_steps = pd.concat(y_test_all_steps)
    if isinstance(best_model, SARIMAX):
        preds_sarima = pd.concat(preds_sarima)
        best_model_rmse = rmse_calc(y_test_all_steps, preds_sarima)
        best_model_normalized_rmse = normalized_rmse(y_test_all_steps, preds_sarima)
        best_model_r2 = r2_score(y_test_all_steps, preds_sarima)
    elif isinstance(best_model, LinearRegression):
        preds_lin = pd.concat(preds_lin)
        best_model_rmse = rmse_calc(y_test_all_steps, preds_lin)
        best_model_normalized_rmse = normalized_rmse(y_test_all_steps, preds_lin)
        best_model_r2 = r2_score(y_test_all_steps, preds_lin)
    elif isinstance(best_model, TBATS):
        preds_tbats = pd.concat(preds_tbats)
        best_model_rmse = rmse_calc(y_test_all_steps, preds_tbats)
        best_model_normalized_rmse = normalized_rmse(y_test_all_steps, preds_tbats)
        best_model_r2 = r2_score(y_test_all_steps, preds_tbats)
    else:
        raise ValueError(f"Unhandled model type: {type(best_model)}")
    print(f"Best Model {forecast_length}: {type(best_model).__name__}")
    print(f"Best Model RMSE {forecast_length}: {best_model_rmse}")
    print(f"Best Model Normalized RMSE {forecast_length}: {best_model_normalized_rmse}")
    print(f"Best Model R2 Score {forecast_length}: {best_model_r2}")

    # Return best_model name and metrics
    return type(best_model).__name__, best_model_rmse, best_model_normalized_rmse, best_model_r2


def naive_models(train_target: pd.Series, 
                    test_target: pd.Series
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Evaluates multiple naive forecasting baselines using lagged target values.

    For each specified lag (1, 24, 72, and 168 time steps), the function creates a naive forecast
    by shifting the full time series. It then calculates forecast performance on the test set 
    using RMSE, normalized RMSE, and R² score.

    This is useful as a benchmarking tool to compare the performance of more complex models
    against simple lag-based baselines.
    """
    
    # Create dictionary of key value pairs comprised of lag lengths as keys, and all of the test set forecasts for that lag length
    lags = [1, 24, 72, 168]
    naive_forecasts = {}
    full_series = pd.concat([train_target, test_target])
    test_start_idx = len(train_target) # Create the index value to track where test forecasts for lag predictors begin
    for lag in lags: # Iterate through lags list and: 
        lagged = full_series.shift(lag) # Get predicted values by shifting each full series value by the given lag length
        forecast = lagged[test_start_idx:] # Assign all test forecasts for given lag to forecast variable
        naive_forecasts[f"lag_{lag}"] = forecast # Populate empty naive_forecasts dictionary by assigning each 
        # key for a given lag its forecast as a value
    
    # Calculate metrics for lag predictors
    metrics = {}
    # Iterate through key value pairs in naive_forecasts dictionary and 
    # make each key in naive_forecasts a key in metrics and assign it
    # a dictionary comprised of calculated metrics for the given forecasts
    for name, forecast in naive_forecasts.items(): 
        metrics[name] = {'rmse': rmse_calc(test_target, forecast), # make each key in naive_forecasts a key in metrics and assign it a dictionary
                            'nrmse': normalized_rmse(test_target, forecast),
                            'r2_score': r2_score(test_target, forecast)}

    # Return the RMSE, NRMSE and R2 score for each lag predictor 
    return (
        metrics['lag_1']['rmse'], metrics['lag_1']['nrmse'], metrics['lag_1']['r2_score'],
        metrics['lag_24']['rmse'], metrics['lag_24']['nrmse'], metrics['lag_24']['r2_score'],
        metrics['lag_72']['rmse'], metrics['lag_72']['nrmse'], metrics['lag_72']['r2_score'],
        metrics['lag_168']['rmse'], metrics['lag_168']['nrmse'], metrics['lag_168']['r2_score']
    )


    
def save_df(best_model_type: str, forecast_length: str,
            mod_rmse: float, mod_nrmse: float, mod_r2: float, 
            lag_168_rmse: float, lag_168_nrmse: float, lag_168_r2:float,
            lag_72_rmse: float = None, lag_72_nrmse: float = None, lag_72_r2: float = None, 
            lag_24_rmse: float = None, lag_24_nrmse: float = None, lag_24_r2: float = None,
            lag_1_rmse: float = None, lag_1_nrmse: float = None, lag_1_r2: float = None) -> None:
    """
    Creates and saves a DataFrame summarizing evaluation metrics for a forecasting model and naive lag baselines.

    The DataFrame includes RMSE, normalized RMSE, and R² scores for the provided model for the given forecast length
    and relevant lag-based predictors (e.g., lag-1, lag-24, lag-72, lag-168).

    The resulting DataFrame is saved as a CSV file named according to the forecast length
    (e.g., `'hour_test_results.csv'`, `'day_test_results.csv'`).
    """

    # Create DataFrame for hour long forecast length
    if forecast_length == 'hour':
        test_results = pd.DataFrame({ 
        f"best_{forecast_length}_model: {best_model_type}": [mod_rmse, mod_nrmse, mod_r2],
        "lag_1": [lag_1_rmse, lag_1_nrmse, lag_1_r2],
        "lag_24": [lag_24_rmse, lag_24_nrmse, lag_24_r2],
        "lag_72": [lag_72_rmse, lag_72_nrmse, lag_72_r2],
        "lag_168": [lag_168_rmse, lag_168_nrmse, lag_168_r2],
        }, index=["rmse", "nrmse", "r2"])
    
    # Create DataFrames for 12_hours or day long forecasts 
    elif forecast_length in ('12_hours', 'day'):
        test_results = pd.DataFrame({
        f"best_{forecast_length}_model: {best_model_type}": [mod_rmse, mod_nrmse, mod_r2],
        "lag_24": [lag_24_rmse, lag_24_nrmse, lag_24_r2],
        "lag_72": [lag_72_rmse, lag_72_nrmse, lag_72_r2],
        "lag_168": [lag_168_rmse, lag_168_nrmse, lag_168_r2],
        }, index=["rmse", "nrmse", "r2"])

    # Create DataFrame for 3_days forecast length
    elif forecast_length == '3_days':
        test_results = pd.DataFrame({
        f"best_{forecast_length}_model: {best_model_type}": [mod_rmse, mod_nrmse, mod_r2],
        "lag_72": [lag_72_rmse, lag_72_nrmse, lag_72_r2],
        "lag_168": [lag_168_rmse, lag_168_nrmse, lag_168_r2],
        }, index=["rmse", "nrmse", "r2"])

    # Create DataFrame for week long forecast length
    elif forecast_length == 'week':
        test_results = pd.DataFrame({
        f"best_{forecast_length}_model: {best_model_type}": [mod_rmse, mod_nrmse, mod_r2],
        "lag_168": [lag_168_rmse, lag_168_nrmse, lag_168_r2],
        }, index=["rmse", "nrmse", "r2"])

    else: # Raise value error forecast_length parameter is passed wrong string value
        raise ValueError(f"Unsupported forecast length: {forecast_length}. "
        "Expected one of: 'hour', '12_hours', 'day', '3_days', 'week'."
        ) 

    # Save Dataframe to flexible file name which is named according to forecast length   
    test_results.to_csv(f'{forecast_length}_test_results.csv', index=True)