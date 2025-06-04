"""
Utility functions for input validation and coordinate handling.
"""
import numpy as np


def validate_coordinates(coords, dim=3):
    """
    Ensure coordinates array is of shape (n_points, dim) and is numeric.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != dim:
        raise ValueError(f"Coordinates must be of shape (n_samples, {dim}), got {coords.shape}")
    return coords


def validate_samples(samples):
    """
    Ensure samples is a 2D array-like: shape (n_samples, n_genes) or (n_samples,) for single gene.
    """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        # convert to (n_samples, 1)
        return samples.reshape(-1, 1)
    if samples.ndim == 2:
        return samples
    raise ValueError(f"Samples must be a 1D or 2D array, got {samples.ndim}D")
