"""
K-fold cross-validation to evaluate interpolation methods using different metrics.
"""
import numpy as np
from sklearn.model_selection import KFold
from .interpolation_core import interpolate
from .metrics import mse, r, r2

def cross_validate(
    samples,
    sample_coords,
    methods=['linear', 'nearest', 'spline', 'exponential', 'knn'],
    k=5,
    random_state=None,
    metrics=['mse'],
    method_kwargs=None
):
    """
    Perform k-fold cross-validation to compare interpolation methods using one or more metrics.

    Parameters
    ----------
    samples : array-like, shape (n_samples,) or (n_samples, n_features)
        Expression values.
    sample_coords : array-like, shape (n_samples, 3)
        Sample coordinates in MNI space.
    methods : list of str
        List of methods to evaluate.
    k : int
        Number of folds.
    random_state : int or None
        Random seed for reproducibility.
    metrics : str or list of str, optional
        One or more metrics: 'mse', 'r', 'r2'.
    method_kwargs : dict or None
        Optional dict mapping method names to dicts of keyword arguments.
        For example: {'exponential': {'epsilon': 0.5}, 'knn': {'n_neighbors': 5, 'weighting': 'power', 'power': 2}}.

    Returns
    -------
    results : dict
        Dictionary mapping each method to a sub-dictionary of average scores for each metric.
        Example: {'linear': {'mse': 0.1, 'r': 0.8}, 'knn': {'mse': 0.09, 'r': 0.82}}.
    """
    samples = np.asarray(samples)
    sample_coords = np.asarray(sample_coords)

    # Ensure metrics is a list
    if isinstance(metrics, str):
        metrics = [metrics]

    # Map metric names to functions
    metric_funcs = {}
    for met in metrics:
        if met == 'mse':
            metric_funcs['mse'] = mse
        elif met == 'r':
            metric_funcs['r'] = r
        elif met == 'r2':
            metric_funcs['r2'] = r2
        else:
            raise ValueError(f"Unknown metric: {met}")

    # Ensure samples is 2D
    if samples.ndim == 2 and samples.shape[1] > 1:
        n_targets = samples.shape[1]
    else:
        n_targets = 1
        samples = samples.reshape(-1, 1)

    # Default empty dict if no method-specific kwargs
    if method_kwargs is None:
        method_kwargs = {}

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Initialize scores: method → metric → list of fold scores
    scores = {method: {met: [] for met in metrics} for method in methods}

    for train_idx, test_idx in kf.split(sample_coords):
        train_coords = sample_coords[train_idx]
        test_coords = sample_coords[test_idx]
        train_vals = samples[train_idx]  # shape (n_train, n_targets)
        test_vals = samples[test_idx]    # shape (n_test, n_targets)

        for method in methods:
            # Extract method-specific kwargs
            m_kwargs = method_kwargs.get(method, {})

            # Interpolate predictions for each target feature
            preds = []
            for t in range(n_targets):
                pred = interpolate(
                    train_vals[:, t], train_coords, test_coords,
                    method=method, **m_kwargs
                )
                preds.append(pred)
            preds = np.vstack(preds).T  # shape (n_test, n_targets)

            # Compute each metric across targets and average
            for met, func in metric_funcs.items():
                per_target_scores = []
                for t in range(n_targets):
                    y_true = test_vals[:, t]
                    y_pred = preds[:, t]
                    per_target_scores.append(func(y_true, y_pred))
                # average across targets
                avg_met_score = float(np.mean(per_target_scores))
                scores[method][met].append(avg_met_score)

    # Compute average across folds for each method and metric
    avg_scores = {
        method: {met: float(np.mean(scores[method][met])) for met in metrics}
        for method in methods
    }
    return avg_scores,scores