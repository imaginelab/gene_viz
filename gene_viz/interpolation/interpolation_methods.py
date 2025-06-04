import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator, Rbf, LinearNDInterpolator, CloughTocher2DInterpolator

from sklearn.neighbors import NearestNeighbors
import torch
import gpytorch

def linear_interpolation(sample_coords, sample_values, eval_coords):
    """
    Linear interpolation using scipy.interpolate.griddata with 'linear' method.
    If NaNs are present in the result, uses nearest neighbor interpolation to fill them.
    """
    linear = griddata(sample_coords, sample_values, eval_coords, method='linear')
    # Identify NaNs
    mask = np.isnan(linear)
    if np.any(mask):
        nearest = griddata(sample_coords, sample_values, eval_coords, method='nearest')
        linear[mask] = nearest[mask]
    return linear


def nearest_neighbor_interpolation(sample_coords, sample_values, eval_coords):
    """
    Nearest neighbor interpolation using scipy.interpolate.NearestNDInterpolator.
    """
    nn = NearestNDInterpolator(sample_coords, sample_values)
    return nn(eval_coords)


def spline_interpolation(sample_coords, sample_values, eval_coords):
    """
    Spline interpolation using Rbf for 3d.
    """
    sample_coords = np.asarray(sample_coords)
    eval_coords = np.asarray(eval_coords)
    # If 3D, use Rbf with 'thin_plate' or 'cubic'
        # Using RBF with thin_plate or cubic basis
    rbf = Rbf(sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2], sample_values, function='thin_plate')
    return rbf(eval_coords[:, 0], eval_coords[:, 1], eval_coords[:, 2])

def thin_plate_interpolation(sample_coords, sample_values, eval_coords, epsilon=1.0, smooth=0.0):
    """
    Thin-plate RBF interpolation with adjustable epsilon and smooth parameters.

    Parameters
    ----------
    sample_coords : array-like, shape (n_samples, 3)
        Coordinates of samples.
    sample_values : array-like, shape (n_samples,)
        Values at sample coordinates.
    eval_coords : array-like, shape (n_eval, 3)
        Coordinates for evaluation.
    epsilon : float
        Length-scale parameter for thin-plate RBF.
    smooth : float
        Smoothing parameter; nonzero values will regularize the fit.

    Returns
    -------
    interp_values : ndarray, shape (n_eval,)
        Interpolated values at eval_coords.
    """
    sample_coords = np.asarray(sample_coords)
    eval_coords = np.asarray(eval_coords)
    x, y, z = sample_coords.T
    # Create thin-plate RBF with smoothness
    rbf = Rbf(x, y, z, sample_values, function='thin_plate',  smooth=smooth)
    return rbf(eval_coords[:, 0], eval_coords[:, 1], eval_coords[:, 2])


def exponential_interpolation(sample_coords, sample_values, eval_coords, epsilon=1.0):
    """
    Exponential interpolation via RBF with exponential kernel.
    """
    sample_coords = np.asarray(sample_coords)
    eval_coords = np.asarray(eval_coords)
    rbf = Rbf(*sample_coords.T, sample_values, function='exponential', epsilon=epsilon)
    return rbf(*eval_coords.T)


def knn_interpolation(sample_coords, sample_values, eval_coords, n_neighbors=10, weighting='power',
                      power=2, **kwargs):
    """
    K-Nearest Neighbors interpolation with configurable weighting functions.

    Parameters
    ----------
    sample_coords : array-like, shape (n_samples, 3)
        Coordinates of sparse samples.
    sample_values : array-like, shape (n_samples,)
        Expression values at sample coordinates.
    eval_coords : array-like, shape (n_eval, 3)
        Coordinates where interpolation is evaluated.
    n_neighbors : int
        Number of nearest neighbors to use.
    weighting : str, optional
        Weighting scheme: 'uniform', 'inverse', 'power', 'gaussian', or 'softmax'.
    **kwargs
        Additional parameters for weighting: 
        - power: exponent for 'power' weighting (default=2)
        - sigma: standard deviation for 'gaussian' weighting (default=1.0)
        - temperature: temperature for 'softmax' weighting (default=1.0)

    Returns
    -------
    interp_values : ndarray, shape (n_eval,)
        Interpolated values at eval_coords.
    """
    sample_coords = np.asarray(sample_coords)
    sample_values = np.asarray(sample_values)
    eval_coords = np.asarray(eval_coords)

    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(sample_coords)
    distances, indices = nbrs.kneighbors(eval_coords)

    # Compute weights based on scheme
    if weighting == 'uniform':
        weights = np.ones_like(distances)
    elif weighting == 'inverse':
        with np.errstate(divide='ignore'):
            weights = 1.0 / distances
        inf_mask = np.isinf(weights)
        if np.any(inf_mask):
            # If distance == 0, give full weight to that neighbor
            weights[inf_mask] = 0
            weights[inf_mask] = inf_mask[inf_mask]
    elif weighting == 'power':
        p = kwargs.get('power', 2)
        with np.errstate(divide='ignore'):
            weights = 1.0 / np.power(distances, p)
        inf_mask = np.isinf(weights)
        if np.any(inf_mask):
            weights[inf_mask] = 0
            weights[inf_mask] = inf_mask[inf_mask]
    elif weighting == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        weights = np.exp(-np.square(distances) / (2 * sigma**2))
    elif weighting == 'softmax':
        temp = kwargs.get('temperature', 1.0)
        logits = -distances / temp
        exp_logits = np.exp(logits)
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Normalize weights
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_sum[weights_sum == 0] = 1  # Avoid division by zero

    # Gather neighbor values and compute weighted average
    values = sample_values[indices]
    weighted_sum = np.sum(weights * values, axis=1)
    interp_values = weighted_sum / weights_sum.ravel()
    return interp_values


#others
# GP Regression interpolation using GPyTorch
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='RBF'):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        if kernel == 'RBF':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel == 'Matern':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def gp_interpolation(sample_coords, sample_values, eval_coords, kernel='RBF', training_iter=50, lr=0.1, use_gpu=False):
    """
    Gaussian Process interpolation using GPyTorch, with MPS support on macOS M1/M2.

    Parameters
    ----------
    sample_coords : array-like, shape (n_samples, 3)
        Coordinates of sparse samples.
    sample_values : array-like, shape (n_samples,)
        Expression values at sample coordinates.
    eval_coords : array-like, shape (n_eval, 3)
        Coordinates where interpolation is evaluated.
    kernel : str, optional
        Kernel type: 'RBF' or 'Matern'. Default: 'RBF'.
    training_iter : int, optional
        Number of training iterations. Default: 50.
    lr : float, optional
        Learning rate for optimizer. Default: 0.1.
    use_gpu : bool, optional
        If True and a GPU or MPS backend is available, run training on that device. Default: False.

    Returns
    -------
    interp_values : ndarray, shape (n_eval,)
        Predicted mean values at eval_coords.
    """
    # Determine device: prioritize MPS on macOS, then CUDA, then CPU
    if use_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Convert to torch tensors
    train_x = torch.tensor(sample_coords, dtype=torch.float32).to(device)
    train_y = torch.tensor(sample_values, dtype=torch.float32).to(device)
    test_x = torch.tensor(eval_coords, dtype=torch.float32).to(device)

    # Likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(train_x, train_y, likelihood, kernel=kernel).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        return observed_pred.mean.to('cpu').numpy()