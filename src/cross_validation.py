# Standard library imports
import os
from typing import Generator, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tbats import TBATS
import statsmodels.api as sm  # Possibly for SARIMA initialization

# Local application imports
from src.features_utils import lag_adjustments
from src.model_io import save_best_model, best_model_selection
from src.models import model_lin, model_tbats, sarima
from src.metrics import rmse_calc


n_splits = 5

def expanding_w_time_series_cv(
    data_length: int, 
    n_splits: int, 
    test_size: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
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

    
def blocked_time_series_cv(
    data_length: int, 
    n_splits: int, 
    test_size: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
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
        train_start = i * step
        train_end = step * i + step
        val_start = train_end
        val_end = val_start + test_size

        if val_end > data_length:
            break # Stop if validation window goes past data end

        train_indices = np.arange(train_start, train_end)
        val_indices = np.arange(val_start, val_end)

        yield train_indices, val_indices


def cross_validation(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    n_splits: int,
    test_size: int,
    forecast_length: int,
    model_save_path: str,
    cv_type: str
) -> None:
    X_train_full = lag_adjustments(forecast_length, X_train_full)

    os.makedirs('cv_rmse_scores/cv_avg_rmse_scores', exist_ok=True)
    os.makedirs('cv_rmse_scores/cv_fold_rmse_scores', exist_ok=True)

    preds_lin_all_folds, preds_sarima_all_folds, preds_tbats_all_folds = [], [], []
    y_test_step_all_folds = []

    # Choose CV generator
    if cv_type == 'blocked':
        cv_generator = blocked_time_series_cv(len(X_train_full), n_splits, test_size)
    else:
        cv_generator = expanding_w_time_series_cv(len(X_train_full), n_splits, test_size)
    for fold, (train_idx, test_idx) in enumerate(cv_generator):
        print(f"\nFold {fold}")

        # Set up base training and test sets
        X_base_train, y_base_train = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_test_full, y_test_full = X_train_full.iloc[test_idx], y_train_full.iloc[test_idx]

        assert len(X_test_full) >= forecast_length, "Test size must be >= forecast_length"
        
        # First prediction at full forecast length
        rolling_steps = len(X_test_full) - forecast_length + 1

        preds_lin_fold, preds_sarima_fold, preds_tbats_fold = [], [], []
        y_test_all_steps = []


        for step in range(rolling_steps):
            # Define rolling train and test windows
            X_train = pd.concat([X_base_train, X_test_full.iloc[:step]])
            y_train = pd.concat([y_base_train, y_test_full.iloc[:step]])

            X_test_step = X_test_full.iloc[step : step + forecast_length]
            y_test_step = y_test_full.iloc[step : step + forecast_length]

            if len(X_test_step) < forecast_length:
                break # not enough data left for full forecast

            model_lin.fit(X_train, y_train)
            model_sarima = sarima(y_train)
            results_sarima = model_sarima.fit()
            results_tbats = model_tbats.fit(y_train)
        
            preds_lin = model_lin.predict(X_test_step)
            preds_sarima = results_sarima.forecast(steps=forecast_length)
            preds_tbats = results_tbats.forecast(steps=forecast_length)

            preds_lin = pd.Series(preds_lin)
            preds_sarima = pd.Series(preds_sarima)
            preds_tbats = pd.Series(preds_tbats)

            preds_lin_fold.append(preds_lin)
            preds_sarima_fold.append(preds_sarima)
            preds_tbats_fold.append(preds_tbats)

            y_test_all_steps.append(y_test_step)

        preds_lin_fold = pd.concat(preds_lin_fold)
        preds_sarima_fold = pd.concat(preds_sarima_fold)
        preds_tbats_fold = pd.concat(preds_tbats_fold)
        y_test_all_steps = pd.concat(y_test_all_steps)

        #save fold averages
        fold_results = pd.DataFrame([{
            'fold': fold,
            'model_lin_rmse': rmse_calc(y_test_all_steps, preds_lin_fold),
            'model_sarima_rmse': rmse_calc(y_test_all_steps, preds_sarima_fold),
            'model_tbats_rmse': rmse_calc(y_test_all_steps, preds_tbats_fold)
        }])
        
        filename = f"cv_rmse_scores/cv_fold_rmse_scores/{cv_type}_cv_fold_{fold}_{forecast_length}h.csv"
        fold_results.to_csv(filename, index=False)

        preds_lin_all_folds.append(preds_lin_fold)
        preds_sarima_all_folds.append(preds_sarima_fold)
        preds_tbats_all_folds.append(preds_tbats_fold)

        y_test_step_all_folds.append(y_test_all_steps)

    #Final evaluation across all folds
    preds_lin_all_folds = pd.concat(preds_lin_all_folds)
    preds_sarima_all_folds = pd.concat(preds_sarima_all_folds)
    preds_tbats_all_folds = pd.concat(preds_tbats_all_folds)

    y_test_step_all_folds = pd.concat(y_test_step_all_folds)

    model_lin_rmse = rmse_calc(y_test_step_all_folds, preds_lin_all_folds)
    model_sarima_rmse = rmse_calc(y_test_step_all_folds, preds_sarima_all_folds)
    model_tbats_rmse = rmse_calc(y_test_step_all_folds, preds_tbats_all_folds)

    print("\n=== RMSE Across All Folds ===")
    print(f"Linear Model RMSE: {model_lin_rmse:.2f}")
    print(f"SARIMA Model RMSE: {model_sarima_rmse:.2f}")
    print(f"TBATS Model RMSE: {model_tbats_rmse:.2f}")

    summary = pd.DataFrame([{
        'model_lin_rmse': model_lin_rmse,
        'model_sarima_rmse': model_sarima_rmse,
        'model_tbats_rmse': model_tbats_rmse,
    }])
            
    summary_file = f"cv_rmse_scores/cv_avg_rmse_scores/{cv_type}_avg_rmse_{forecast_length}h.csv"
    summary.to_csv(summary_file, index=False)

    # Select and save best model
    rmse_list = [model_lin_rmse, model_sarima_rmse, model_tbats_rmse]
    model_list = [model_lin, model_sarima, model_tbats]
    best_model = best_model_selection(rmse_list, model_list)
    print("Best model object type:", type(best_model).__name__)
    save_best_model(best_model, model_save_path)
