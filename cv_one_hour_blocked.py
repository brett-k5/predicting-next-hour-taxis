# Standard library imports
import random

# Third-party imports
import numpy as np
from sklearn.metrics import mean_squared_error

# Local application imports
from src.cross_validation import cross_validation, n_splits
from src.models import model_lin, model_tbats, sarima
from src.pre_processing import X_train_hour, y_train_hour


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    np.random.seed(12345)
    random.seed(12345)

    cross_validation(X_train_hour,
                     y_train_hour,
                     n_splits,
                     4,
                     1,
                     'models/best_model_hour_blocked.pkl',
                     'blocked')