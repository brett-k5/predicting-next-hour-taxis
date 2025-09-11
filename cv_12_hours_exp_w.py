# Standard library imports
import random

# Third-party imports
import numpy as np
from sklearn.metrics import mean_squared_error

# Local application imports
from src.cross_validation import cross_validation, n_splits
from src.models import model_lin, model_tbats, sarima
from src.pre_processing import X_train_12_hours, y_train_12_hours



if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    np.random.seed(12345)
    random.seed(12345)

    cross_validation(X_train_12_hours,
                     y_train_12_hours,
                     n_splits,
                     15,
                     12,
                     'models/best_model_12_hours_exp_w.pkl',
                     'expanded_w')