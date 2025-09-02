
import pandas as pd
import statistics as st
# Third-party imports
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.linear_model import LinearRegression
from tbats import TBATS
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Local application imports
from src.metrics import rmse_calc, normalized_rmse


def test(best_model, 
         X_train: pd.DataFrame, 
         y_train: pd.Series, 
         X_test: pd.DataFrame, 
         y_test: pd.Series,
         forecast_length: str,
         forecast_length_int: int) -> tuple[str, float, float, float]:
    preds_sarima, preds_lin, preds_tbats = [], [], []
    y_test_all_steps = []
    rolling_steps = len(X_test) - forecast_length_int + 1
    for step in range(rolling_steps):

        print(f"\n\nStep Number: {step}\n\n")
        # Define rolling train and test windows
        X_train = pd.concat([X_train, X_test.iloc[:step]])
        y_train = pd.concat([y_train, y_test.iloc[:step]])

        X_test_step = X_test.iloc[step : step + forecast_length_int]
        y_test_step = y_test.iloc[step : step + forecast_length_int]

        if len(X_test_step) < forecast_length_int:
            break # not enough data left for full forecast

        y_test_all_steps.append(y_test_step)

        if isinstance(best_model, SARIMAXResults):
            best_model = SARIMAX(y_train,
                                order=best_model.specification["order"],
                                seasonal_order=best_model.specification["seasonal_order"],
                                enforce_stationarity=best_model.specification["enforce_stationarity"],
                                enforce_invertibility=best_model.specification["enforce_invertibility"])
            results = best_model.fit()
            preds = results.forecast(steps=forecast_length_int)
            preds = pd.Series(preds)
            preds_sarima.append(preds)
        elif isinstance(best_model, LinearRegression):
            best_model.fit(X_train, y_train)
            preds = best_model.predict(X_test_step)
            preds = pd.Series(preds)
            preds_lin.append(preds)
        elif isinstance(best_model, TBATS):
            results = best_model.fit(y_train)
            preds = results.forecast(steps=forecast_length_int)
            preds = pd.Series(preds)
            preds_tbats.append(preds)
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
    return type(best_model).__name__, best_model_rmse, best_model_normalized_rmse, best_model_r2


def naive_models(train_target: pd.Series, 
                    test_target: pd.Series
) -> tuple[float, float, float, 
float, float, float, 
float, float, float, 
float, float, float]:
    lags = [1, 24, 72, 168]
    naive_forecasts = {}
    full_series = pd.concat([train_target, test_target])
    test_start_idx = len(train_target)

    for lag in lags:
        lagged = full_series.shift(lag)
        forecast = lagged[test_start_idx:]
        naive_forecasts[f"lag_{lag}"] = forecast
    metrics = {}
    for name, forecast in naive_forecasts.items():
        metrics[name] = {'rmse': rmse_calc(test_target, forecast),
                            'nrmse': normalized_rmse(test_target, forecast),
                            'r2_score': r2_score(test_target, forecast)}
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
    if forecast_length == 'hour':
        test_results = pd.DataFrame({
        f"best_{forecast_length}_model: {best_model_type}": [mod_rmse, mod_nrmse, mod_r2],
        "lag_1": [lag_1_rmse, lag_1_nrmse, lag_1_r2],
        "lag_24": [lag_24_rmse, lag_24_nrmse, lag_24_r2],
        "lag_72": [lag_72_rmse, lag_72_nrmse, lag_72_r2],
        "lag_168": [lag_168_rmse, lag_168_nrmse, lag_168_r2],
        }, index=["rmse", "nrmse", "r2"])
    elif forecast_length in ('12_hours', 'day'):
        test_results = pd.DataFrame({
        f"best_{forecast_length}_model: {best_model_type}": [mod_rmse, mod_nrmse, mod_r2],
        "lag_24": [lag_24_rmse, lag_24_nrmse, lag_24_r2],
        "lag_72": [lag_72_rmse, lag_72_nrmse, lag_72_r2],
        "lag_168": [lag_168_rmse, lag_168_nrmse, lag_168_r2],
        }, index=["rmse", "nrmse", "r2"])
    elif forecast_length == '3_days':
        test_results = pd.DataFrame({
        f"best_{forecast_length}_model: {best_model_type}": [mod_rmse, mod_nrmse, mod_r2],
        "lag_72": [lag_72_rmse, lag_72_nrmse, lag_72_r2],
        "lag_168": [lag_168_rmse, lag_168_nrmse, lag_168_r2],
        }, index=["rmse", "nrmse", "r2"])
    elif forecast_length == 'week':
        test_results = pd.DataFrame({
        f"best_{forecast_length}_model: {best_model_type}": [mod_rmse, mod_nrmse, mod_r2],
        "lag_168": [lag_168_rmse, lag_168_nrmse, lag_168_r2],
        }, index=["rmse", "nrmse", "r2"])
    else:
        raise ValueError(f"Unsupported forecast length: {forecast_length}. "
        "Expected one of: 'hour', '12_hours', 'day', '3_days', 'week'."
        ) 
        
    test_results.to_csv(f'{forecast_length}_test_results.csv', index=True)