# Standard library imports
import os
import pickle
import warnings

# Typing imports
from typing import Union, Optional

# Third-party library imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from tbats import TBATS
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


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



def load_model(best_model_path: str) -> Union[
      SARIMAXResults, LinearRegression, TBATS]:
    with open(best_model_path, "rb") as f:
            best_model = pickle.load(f)
    return best_model



def rmse_comp(df_blocked: pd.DataFrame, 
              df_exp_w: pd.DataFrame, 
              forecast_length: str, 
              forecast_length_int: int,
              override: bool = False,
              override_model: Optional[Union[SARIMAXResults, LinearRegression, TBATS]] = None) -> Union[
      SARIMAXResults, 
      LinearRegression, 
      TBATS]:
    df_blocked = df_blocked.T
    df_blocked.columns = ['rmse']
    df_exp_w = df_exp_w.T
    df_exp_w.columns = ['rmse']
    if df_blocked['rmse'].idxmin() == df_exp_w['rmse'].idxmin():
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
            blocked_model = load_model(f'models/best_model_{forecast_length}_blocked.pkl')
            if isinstance(blocked_model, override_model):
                # Return the model that performed best on blocked CV 
                return blocked_model
            else:
                # Return the model that performed best on expanding window CV 
                return load_model(f'models/best_model_{forecast_length}_exp_w.pkl')
        else:
            print('Review average rmse scores and fold rmse scores to make sure you agree with the selection.')
            print(f'{forecast_length}')
            print(f'Blocked Cross Validation Results:\n{df_blocked}')
            print(f'Expanding Window Cross Validation Results:\n{df_exp_w}')
            fold_rmse_dfs = {}
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
            for var_name, file_path in folds_rmse.items():
                fold_rmse_dfs[var_name] = pd.read_csv(file_path)
            all_folds_df = pd.concat(fold_rmse_dfs.values(), axis=0, ignore_index=True)
            all_folds_df = all_folds_df.set_index('fold')
            print(all_folds_df)
            per_model_fold_std = all_folds_df.std()
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
            