# Standard library imports
import os
import pickle
import warnings

# Typing imports
from typing import Any, List, Optional, Union

# Third-party library imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from tbats import TBATS
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def best_model_selection(rmse_list, model_list):
    """
    Returns the model corresponding to the lowest RMSE value.

    Raises:
        ValueError: If rmse_list and model_list are not the same length.
    """
    if len(rmse_list) != len(model_list):
        raise ValueError("rmse_list and model_list must be of the same length.")
    min_index = rmse_list.index(min(rmse_list))
    best_model = model_list[min_index]
    return best_model


def save_best_model(best_model, model_save_path):
    """
    Saves a serializable specification of the best model to disk.

    Currently supports SARIMAXResults, LinearRegression, and TBATS models.
    Only model specifications are saved (not the fitted instances), allowing refitting later.

    If the model type is not recognized, no file is saved and a warning is printed.
    """
    os.makedirs('models', exist_ok=True)
    if isinstance(best_model, SARIMAXResults):
        # Save only the model specification (SARIMAX class + data passed to initializer)
        # so we can refit it later
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



def load_model(best_model_path: str) -> Union[SARIMAXResults, LinearRegression, TBATS]:
    """
    Loads a saved model specification from disk.

    Raises:
        FileNotFoundError: If the file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled properly.
    """
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"No such file: '{best_model_path}'")
    try:
        with open(best_model_path, "rb") as f:
            best_model = pickle.load(f)
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Failed to unpickle model from '{best_model_path}': {e}")

    return best_model



def rmse_comp(df_blocked: pd.DataFrame, 
              df_exp_w: pd.DataFrame, 
              forecast_length: str, 
              forecast_length_int: int,
              override: bool = False,
              override_model: Optional[Union[SARIMAXResults, LinearRegression, TBATS]] = None) -> Union[SARIMAXResults, LinearRegression, TBATS]:
    """
    Compare blocked and expanding window cross-validation RMSEs and return the best model specification.

    The function selects the model with the lowest RMSE during both expanded_w or blocked cv,
    or, if the lowest RMSE on expanded_w and blocked cv are not the same and override is True,
    returns the model matching the override_model type. If override is False and lowest RMSEs
    are different between cv methods disagree model selection is set to return the model with
    the lowest std across all folds and cv_methods by default.
    """

    # Transpose df_blocked and df_exp so idxmin() 
    # can be used to select lowest rmse model
    df_blocked = df_blocked.T
    df_blocked.columns = ['rmse']
    df_exp_w = df_exp_w.T
    df_exp_w.columns = ['rmse']

    if df_blocked['rmse'].idxmin() == df_exp_w['rmse'].idxmin():
        # print warning message if override parameter set to True when it shouldn't be
        # (i.e. when expanded_w and blocked cv agree on the best model) 
        if override:
            print(f"For {forecast_length} model:")
            warnings.warn(
            "`override=True` when the minimum rmse on blocked and exp_w CV are the same may "
            "lead to misleading results. Consider using the default override=False.",
            UserWarning
            )
        return load_model(f'models/best_model_{forecast_length}_blocked.pkl')

    else:
        if override:
            # select blocked_model and override_model are the same, and select best exp_w model when 
            blocked_model = load_model(f'models/best_model_{forecast_length}_blocked.pkl')
            exp_w_model = load_model(f'models/best_model_{forecast_length}_exp_w.pkl')
            if isinstance(blocked_model, override_model):
                return blocked_model
            elif isinstance(exp_w_model, override_model):
                return exp_w_model
            else:
                raise TypeError("override_model must be of the supported types: "
                                "LinearRegression, SARIMAXResults, or TBATS instance.")
        else:
            # Print User instructions
            print('Review average rmse scores and fold rmse scores to make sure you agree with the selection.')
            print(f'{forecast_length}')
            print(f'Blocked Cross Validation Results:\n{df_blocked}')
            print(f'Expanding Window Cross Validation Results:\n{df_exp_w}')

            # Initialize empty dictionary
            fold_rmse_dfs = {}

            # Create dictionary with flexible variable names as keys and file paths as as values
            folds_rmse = {f'blocked_fold_0_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/blocked_cv_fold_0_{forecast_length_int}h.csv',
                        f'exp_w_cv_fold_0_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/expanded_w_cv_fold_0_{forecast_length_int}h.csv',
                        f'blocked_fold_1_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/blocked_cv_fold_1_{forecast_length_int}h.csv',
                        f'exp_w_cv_fold_1_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/expanded_w_cv_fold_1_{forecast_length_int}h.csv',
                        f'blocked_fold_2_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/blocked_cv_fold_2_{forecast_length_int}h.csv',
                        f'exp_w_cv_fold_2_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/expanded_w_cv_fold_2_{forecast_length_int}h.csv',
                        f'blocked_fold_3_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/blocked_cv_fold_3_{forecast_length_int}h.csv',
                        f'exp_w_cv_fold_3_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/expanded_w_cv_fold_3_{forecast_length_int}h.csv',
                        f'blocked_fold_4_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/blocked_cv_fold_4_{forecast_length_int}h.csv',
                        f'exp_w_cv_fold_4_{forecast_length}': f'cv_rmse_scores/cv_fold_rmse_scores/expanded_w_cv_fold_4_{forecast_length_int}h.csv'}
            
            # Iterate through folds_rmse dictionary 
            for var_name, file_path in folds_rmse.items():
                # assign dictionary key pd.DataFrame saved at its value (which is a file path)
                fold_rmse_dfs[var_name] = pd.read_csv(file_path) 
            
            # Concatinate all DataFrames loaded in the above for loop
            all_folds_df = pd.concat(fold_rmse_dfs.values(), axis=0, ignore_index=True)
            all_folds_df = all_folds_df.set_index('fold') # Set the index of all_folds_df to match the 'folds' column
            print(all_folds_df) # print DataFrame containing all folds cv rmse scores for both cv types

            # Calculate std for all folds across both cv methods
            per_model_fold_std = all_folds_df.std()

            # Return the model with the lowest std across all cv methods and folds with idxmin()
            lowest_std_model = per_model_fold_std.idxmin()
            if lowest_std_model == 'model_lin_rmse':
                suffix = 'blocked' if df_blocked['rmse'].idxmin() == 'model_lin_rmse' else 'exp_w'
                return load_model(f'models/best_model_{forecast_length}_{suffix}.pkl')
            if lowest_std_model == 'model_sarima_rmse':
                suffix = 'blocked' if df_blocked['rmse'].idxmin() == 'model_sarima_rmse' else 'exp_w'
                return load_model(f'models/best_model_{forecast_length}_{suffix}.pkl')
            if lowest_std_model == 'model_tbats_rmse':
                suffix = 'blocked' if df_blocked['rmse'].idxmin() == 'model_tbats_rmse' else 'exp_w'
                return load_model(f'models/best_model_{forecast_length}_{suffix}.pkl')
            raise ValueError(f"Unexpected model type: {lowest_std_model}")
            