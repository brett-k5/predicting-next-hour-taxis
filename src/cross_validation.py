# Standard library imports
import os
from typing import Generator, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tbats import TBATS
import statsmodels.api as sm  

# Local application imports
from src.model_io import save_best_model, best_model_selection
from src.models import model_lin, model_tbats, sarima
from src.metrics import rmse_calc

# Number of folds to be used for all models and cv methods
n_splits = 5

def expanding_w_time_series_cv(
    data_length: int, 
    n_splits: int, 
    test_size: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generates train/validation indices for expanding window cross-validation.

    Parameters:
    - data_length: int, length of your time series data
    - n_splits: int, number of folds
    - test_size: int, size of each validation block

    Yields:
    - train_indices: np.array of training indices
    - val_indices: np.array of validation indices
    """
    train_size = (data_length - test_size * n_splits) // n_splits # Calculate the size of each expanding window step
    for i in range(n_splits):
        # end training at the index given by the all sum of all previous training and 
        train_end = (train_size + test_size) * i + train_size # validation lengths plus train_size for the current loop
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
    train_size = (data_length - test_size * n_splits) // n_splits # Calculate the size of the training set for each fold
    for i in range(n_splits):
        if i == 0: # start training at index 0 for first loop
            train_start = i # Unlike in expanding window cv, the starting point for training has to move along with train_end
        else:
            # Start training on loops > 0 at the index following the final sample from the previous round's validation set
            train_start = (i * (train_size + test_size)) 
            
        # End training at the index given by the length of all prior rounds' 
        train_end = (train_size + test_size) * i + train_size # training and validation sets plus this round's training set
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
    """
    Performs cross-validation using either blocked or expanding window strategies,
    and evaluates three forecasting models (Linear, SARIMA, TBATS) using RMSE.

    === Code Logic Summary ===
    For each cross-validation fold, the function applies a rolling forecast strategy.

    At each rolling step:
    - One new data point is added to the training set.
    - The model is retrained using this expanded training set.
    - The model then predicts the next `forecast_length` target values, starting from the time point
      immediately following the most recent training observation.

    This process repeats for:
        rolling_steps = len(test set) - forecast_length + 1

    At each step, predictions and the corresponding true target values are stored as pandas Series.

    Once all steps are complete within a fold:
    - The predictions for each model are concatenated.
    - They are compared to the full set of step-wise true values using a custom `rmse_calc()` function.
    - The RMSE scores are saved per fold to:
        'cv_rmse_scores/cv_fold_rmse_scores/'

    After all folds are processed:
    - All fold-level predictions and true values are concatenated across folds.
    - RMSE scores for each model are calculated across all folds.
    - These scores are saved to:
        'cv_rmse_scores/cv_all_folds_rmse_scores/'

    Finally, the best-performing model (based on lowest RMSE) is selected and saved to disk using
    the `save_best_model()` function.

    Note:
        - `forecast_length` must not exceed the size of the test set (`test_size`).
        - Custom model training and evaluation utilities such as `sarima`, `rmse_calc`, and
          `save_best_model` are assumed to be defined externally.
    """
    
    # Create directories to store avg cv results across folds and per fold cv results
    os.makedirs('cv_rmse_scores/cv_all_folds_rmse_scores', exist_ok=True)
    os.makedirs('cv_rmse_scores/cv_fold_rmse_scores', exist_ok=True)

    # Initiate empty lists for predictions and and true target values through all cv folds
    preds_lin_all_folds, preds_sarima_all_folds, preds_tbats_all_folds = [], [], []
    y_test_step_all_folds = [] 

    # Generate cv index with chosen generator
    if cv_type == 'blocked':
        cv_generator = blocked_time_series_cv(len(X_train_full), n_splits, test_size)
    elif cv_type == 'expanded_w':
        cv_generator = expanding_w_time_series_cv(len(X_train_full), n_splits, test_size)
    else:
        raise ValueError("model_type parameter was passed an unexpected string value. "
                         "model_type must be passed either 'blocked' or 'expanded_w'.")
    for fold, (train_idx, test_idx) in enumerate(cv_generator):
        print(f"\nFold {fold}")

        # Set up base training and test sets
        X_base_train, y_base_train = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_test_full, y_test_full = X_train_full.iloc[test_idx], y_train_full.iloc[test_idx]

        assert len(X_test_full) >= forecast_length, "Test size must be >= forecast_length"
        
        # Calculate number of steps that will be added to training set during each fold
        rolling_steps = len(X_test_full) - forecast_length + 1
        
        # Initiate empty lists for per fold predictions and target_values 
        preds_lin_fold, preds_sarima_fold, preds_tbats_fold = [], [], []
        y_test_all_steps = [] # y_test_all_steps is necessary because intra fold target values will be different
        # as the loop iterates through rolling steps.

        # Each step represents one data point added to the training set for a given fold
        # The model is retrained and returns a new set of updated predictions 
        # with each data point added to the training set
        for step in range(rolling_steps): # Iterate through rolling steps on each fold

            # Define rolling train and test windows
            X_train = pd.concat([X_base_train, X_test_full.iloc[:step]])
            y_train = pd.concat([y_base_train, y_test_full.iloc[:step]])
            X_test_step = X_test_full.iloc[step : step + forecast_length]
            y_test_step = y_test_full.iloc[step : step + forecast_length]

            if len(X_test_step) < forecast_length:
                break # not enough data left for full forecast

            model_lin.fit(X_train, y_train)
            # sarima() is a custom function that returns fitted sarima model. This is necessary because SARIMAX requires y_train
            model_sarima = sarima(y_train) # as input to be initialized. See models.py script for more details
            results_sarima = model_sarima.fit()
            results_tbats = model_tbats.fit(y_train)
        
            preds_lin = model_lin.predict(X_test_step) # returns np.ndarray
            preds_sarima = results_sarima.forecast(steps=forecast_length) # returns pd.Series
            preds_tbats = results_tbats.forecast(steps=forecast_length) # returns np.ndarray
            
            # Ensure all sets of predictions are pandas series 
            preds_lin = pd.Series(preds_lin) 
            preds_sarima = pd.Series(preds_sarima)
            preds_tbats = pd.Series(preds_tbats)
            
            # Append empty per fold prediction lists with pd.Series of predictions for current rolling step
            preds_lin_fold.append(preds_lin)
            preds_sarima_fold.append(preds_sarima)
            preds_tbats_fold.append(preds_tbats)
            
            # Append empty per fold true target values with pd.Series of true target value for current rolling step
            y_test_all_steps.append(y_test_step)

        # concatinate all pd.Series contained in each model's all rolling steps predictions list for each list/model
        preds_lin_fold = pd.concat(preds_lin_fold)
        preds_sarima_fold = pd.concat(preds_sarima_fold)
        preds_tbats_fold = pd.concat(preds_tbats_fold)

        # concatinate all pd.Series contained in all rolling steps true target values list 
        y_test_all_steps = pd.concat(y_test_all_steps)

        # Save per fold rmse scores
        fold_results = pd.DataFrame([{
            'fold': fold,
            'model_lin_rmse': rmse_calc(y_test_all_steps, preds_lin_fold),
            'model_sarima_rmse': rmse_calc(y_test_all_steps, preds_sarima_fold),
            'model_tbats_rmse': rmse_calc(y_test_all_steps, preds_tbats_fold)
        }])
        
        # Create flexible file path for the DataFrame created for each fold's per model rmse scores
        filename = f"cv_rmse_scores/cv_fold_rmse_scores/{cv_type}_cv_fold_{fold}_{forecast_length}h.csv"
        fold_results.to_csv(filename, index=False)
        
        # Append each empty all folds predictions list with pd.Series of predictions for current fold for each model
        preds_lin_all_folds.append(preds_lin_fold)
        preds_sarima_all_folds.append(preds_sarima_fold)
        preds_tbats_all_folds.append(preds_tbats_fold)
        
        # Append each empty all folds true target values list with pd.Series of true target values for current fold
        y_test_step_all_folds.append(y_test_all_steps)

    # Concatinate all pd.Series in all folds predictions list for each model
    preds_lin_all_folds = pd.concat(preds_lin_all_folds)
    preds_sarima_all_folds = pd.concat(preds_sarima_all_folds)
    preds_tbats_all_folds = pd.concat(preds_tbats_all_folds)

    # Concatinate all pd.Series in all folds true target values list 
    y_test_step_all_folds = pd.concat(y_test_step_all_folds)
    
    # Calculate rmse scores for each model across all folds
    model_lin_rmse = rmse_calc(y_test_step_all_folds, preds_lin_all_folds)
    model_sarima_rmse = rmse_calc(y_test_step_all_folds, preds_sarima_all_folds)
    model_tbats_rmse = rmse_calc(y_test_step_all_folds, preds_tbats_all_folds)
    
    # Print rmse results for each model across all folds
    print("\n=== RMSE Across All Folds ===")
    print(f"Linear Model RMSE: {model_lin_rmse:.2f}")
    print(f"SARIMA Model RMSE: {model_sarima_rmse:.2f}")
    print(f"TBATS Model RMSE: {model_tbats_rmse:.2f}")
    
    # Create pd.DataFrame containing rmse results for each model across all folds
    summary = pd.DataFrame([{
        'model_lin_rmse': model_lin_rmse,
        'model_sarima_rmse': model_sarima_rmse,
        'model_tbats_rmse': model_tbats_rmse,
    }])

    # Create flexible file path for each model's rmse scores across all folds       
    summary_file = f"cv_rmse_scores/cv_all_folds_rmse_scores/{cv_type}_all_folds_rmse_{forecast_length}h.csv"
    summary.to_csv(summary_file, index=False)

    # Select and save best model
    rmse_list = [model_lin_rmse, model_sarima_rmse, model_tbats_rmse]
    model_list = [model_lin, model_sarima, model_tbats]
    best_model = best_model_selection(rmse_list, model_list) # See model_io.py for best_model_selection() code
    print("Best model object type:", type(best_model).__name__)
    save_best_model(best_model, model_save_path) # See model_io.py for save_best_model() code 
