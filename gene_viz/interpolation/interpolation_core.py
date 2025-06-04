import numpy as np
from .interpolation_utils import validate_coordinates, validate_samples
from .interpolation_methods import (
    linear_interpolation,
    nearest_neighbor_interpolation,
    spline_interpolation,
    exponential_interpolation,
    thin_plate_interpolation,
    knn_interpolation,
)


def interpolate(
    samples,
    sample_coords,
    eval_coords,
    method='linear',
    **kwargs
):
    """
    Interpolate sample values at evaluation coordinates.

    Parameters
    ----------
    samples : array-like, shape (n_samples,) or (n_samples, n_features)
        Expression values for each sample. Can be multiple genes at once.
    sample_coords : array-like, shape (n_samples, 3)
        Coordinates of samples in MNI space.
    eval_coords : array-like, shape (n_eval, 3)
        Coordinates where interpolation is evaluated.
    method : str, optional
        Interpolation method: 'linear', 'nearest', 'spline', 'exponential', 'thin_plate', or 'knn'. Default is 'linear'.
    **kwargs : additional keyword arguments to pass to the method-specific function.

    Returns
    -------
    interp_values : ndarray, shape (n_eval, n_features)
        Interpolated expression values at evaluation coordinates.
    """
    # Validate inputs
    sample_coords = validate_coordinates(sample_coords, dim=3)
    eval_coords = validate_coordinates(eval_coords, dim=3)
    samples = validate_samples(samples)

    # Dispatch based on method
    method = method.lower()
    if method == 'linear':
        func = linear_interpolation
    elif method in ('nearest', 'nearest_neighbor'):
        func = nearest_neighbor_interpolation
    elif method == 'spline':
        func = spline_interpolation
    elif method == 'exponential':
        func = exponential_interpolation
    elif method == 'thin_plate':
        func = thin_plate_interpolation
    elif method == 'knn':
        func = knn_interpolation
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # If multiple features (genes), interpolate each separately
    if samples.ndim == 2 and samples.shape[1] > 1:
        results = []
        for i in range(samples.shape[1]):
            vals = func(sample_coords, samples[:, i], eval_coords, **kwargs)
            results.append(vals)
        return np.vstack(results).T
    else:
        sample_values = samples.flatten()
        return func(sample_coords, sample_values, eval_coords, **kwargs)

