"""
Various evaluation metrics for interpolation.
"""
import numpy as np
from scipy.stats import pearsonr


def mse(y_true, y_pred):
    """
    Mean squared error between true and predicted values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def r(y_true, y_pred):
    """
    Pearson correlation coefficient between true and predicted values.
    Returns correlation coefficient (not p-value).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # If constant array, pearsonr will fail; handle that
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return 0.0
    return pearsonr(y_true, y_pred)[0]


def r2(y_true, y_pred):
    """
    Coefficient of determination (R^2) between true and predicted values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot
