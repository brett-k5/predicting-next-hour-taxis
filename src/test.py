
import pandas as pd
import statistics as st
# Third-party imports
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.linear_model import LinearRegression
from tbats import TBATS
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Local application imports
from src.features_utils import lag_adjustments
from src.metrics import rmse_calc, normalized_rmse



def test(best_model, 
         X_train: pd.DataFrame, 
         y_train: pd.Series, 
         X_test: pd.DataFrame, 
         y_test: pd.Series,
         forecast_length: str,
         forecast_length_int: int):
    preds_sarima, preds_lin, preds_tbats = [], [], []
    y_test_all_steps = []
    rolling_steps = len(X_test) - forecast_length_int + 1
    for step in range(rolling_steps):

        print(f"\n\nStep Number: {step}\n\n")
        # Define rolling train and test windows
        X_train = pd.concat([X_train, X_test.iloc[:step]])
        y_train = pd.concat([y_train, y_test.iloc[:step]])
        
        print(f"length of y_test: {len(y_test)}")
        
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
            X_train = lag_adjustments(forecast_length_int, X_train)
            X_test_step = lag_adjustments(forecast_length_int, X_test_step)
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
    else:
        preds_tbats = pd.concat(preds_tbats)
        best_model_rmse = rmse_calc(y_test_all_steps, preds_tbats)
        best_model_normalized_rmse = normalized_rmse(y_test_all_steps, preds_tbats)
        best_model_r2 = r2_score(y_test_all_steps, preds_tbats)
    print(f"Best Model {forecast_length}: {type(best_model).__name__}")
    print(f"Best Model RMSE {forecast_length}: {best_model_rmse}")
    print(f"Best Model Normalized RMSE {forecast_length}: {best_model_normalized_rmse}")
    print(f"Best Model R2 Score {forecast_length}: {best_model_r2}")
    return type(best_model).__name__, best_model_rmse, best_model_normalized_rmse, best_model_r2