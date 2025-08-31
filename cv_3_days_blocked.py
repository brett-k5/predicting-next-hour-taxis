# Standard library imports
import random

# Third-party imports
import numpy as np
from sklearn.metrics import mean_squared_error

# Local application imports
from src.cross_validation import n_splits, cross_validation
from src.models import model_lin, sarima, model_tbats
from src.pre_processing import X_train_3_days, y_train_3_days


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    np.random.seed(12345)
    random.seed(12345)

    cross_validation(X_train_3_days,
                     y_train_3_days,
                     n_splits,
                     75,
                     72,
                     'models/best_model_3_days_blocked.pkl',
                     'blocked')
    