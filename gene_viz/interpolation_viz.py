import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def make_3d_interpolated_grid_mni(samples, sample_coords, interp_func, mni_img, resolution=1):
    """
    Generate a 3D grid of interpolated gene expression values aligned to MNI image.

    Parameters:
        interp_func (callable): Interpolator function from interpolate_expression_3d
        mni_img (nib.Nifti1Image): Loaded MNI image with affine
        resolution (int): Step size in voxel units (1 = full resolution)

    Returns:
        grid_values (np.ndarray): 3D array of interpolated expression values
        X, Y, Z (np.ndarray): 3D meshgrid arrays in MNI space (millimeters)
    """

    affine = mni_img.affine
    shape = mni_img.shape

    # Create grid of voxel indices
    i = np.arange(0, shape[0], resolution)
    j = np.arange(0, shape[1], resolution)
    k = np.arange(0, shape[2], resolution)

    I, J, K = np.meshgrid(i, j, k, indexing='ij')

    # Flatten and convert voxel indices to world (MNI) coordinates
    vox_coords = np.column_stack((I.ravel(), J.ravel(), K.ravel(), np.ones(I.size)))
    world_coords = vox_coords @ affine.T
    world_coords = world_coords[:, :3]  # Drop the homogeneous coordinate

    """
    def interpolate(
    samples,
    sample_coords,
    eval_coords,
    method='knn',
    **kwargs
    ):
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
        Interpolation method: 'linear', 'nearest', 'spline', 'exponential', 'thin_plate', 'knn'. Default is 'linear'.
    **kwargs : additional keyword arguments to pass to the method-specific function.
    Returns
    -------
    interp_values : ndarray, shape (n_eval, n_features)
        Interpolated expression values at evaluation coordinates.
    """
    # Interpolate at world coordinates
    values = interp_func(samples, sample_coords, world_coords, method='knn')

    # Reshape to grid
    grid_shape = (len(i), len(j), len(k))
    grid_values = values.reshape(grid_shape)

    # Also reshape world coordinate grids for optional visualization
    X = world_coords[:, 0].reshape(grid_shape)
    Y = world_coords[:, 1].reshape(grid_shape)
    Z = world_coords[:, 2].reshape(grid_shape)

    return grid_values, X, Y, Z

def plot_volumetrics_plane_alpha(gene_name, mni_volume, gene_vols, section=60, orientation=2, alpha_mask=None):

    if alpha_mask is None:
        # mask gene expression for visualisation
        # Set to NaN where mni_volume is 0
        gene_vols[mni_volume == 0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(30, 30))
        axes[1].set_title(f'Spatial bulk expression: {gene_name}', fontsize=20)

        a = orientation

        mni_section = np.flipud(mni_volume.take(section, axis=a).T)
        gene_im = np.flipud(gene_vols.take(section, axis=a).T)
        gene_im_mask = np.ma.masked_array(gene_im, gene_im == 0)

        axes[0].imshow(mni_section, cmap='Greys_r', vmin=100, vmax=255)
        axes[0].axis('off')

        axes[1].imshow(mni_section, cmap='Greys_r')
        axes[1].imshow(gene_im_mask, cmap='turbo')
        axes[1].axis('off')

        axes[2].imshow(mni_section, cmap='Greys_r', vmin=100, vmax=255)
        axes[2].imshow(gene_im_mask, cmap='turbo', alpha=0.5)
        axes[2].axis('off')

        fig.tight_layout()
        return fig
    
    else:

        # Mask gene expression where MNI volume is zero
        gene_vols[mni_volume == 0] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # One row of 3 plots
        axes[1].set_title(f'Spatial bulk expression: {gene_name}', fontsize=20)

        a = orientation  # Axis for slicing

        # Extract slices and flip for display
        mni_section = np.flipud(mni_volume.take(section, axis=a).T)
        gene_im = np.flipud(gene_vols.take(section, axis=a).T)
        gene_im_mask = np.ma.masked_array(gene_im, np.isnan(gene_im))

        # Prepare alpha slice if provided
        if alpha_mask is not None:
            alpha_slice = np.flipud(alpha_mask.take(section, axis=a).T)
            # Mask alpha wherever gene expression is NaN
            alpha_slice = np.ma.masked_array(alpha_slice, np.isnan(gene_im))
        else:
            alpha_slice = 0.5  # Default fixed transparency

        alpha_slice_mri = alpha_slice * 0.66 # So we can see the mri

        # Plot MNI background only
        axes[0].imshow(mni_section, cmap='Greys_r', vmin=100, vmax=255)
        axes[0].axis('off')

        # Plot overlay without transparency
        #axes[1].imshow(mni_section, cmap='Greys_r')
        axes[1].imshow(gene_im_mask, cmap='turbo', alpha=alpha_slice)
        axes[1].axis('off')

        # Plot overlay with variable transparency
        axes[2].imshow(mni_section, cmap='Greys_r', vmin=100, vmax=255)
        axes[2].imshow(gene_im_mask, cmap='turbo', alpha=alpha_slice_mri)
        axes[2].axis('off')

        fig.tight_layout()
        return fig
    
def get_point_density(coords,search_radius,search_k,mni_img,resolution=1):
    """
    Generate a 3D grid of interpolated gene expression values aligned to MNI image.

    Parameters:
        interp_func (callable): Interpolator function from interpolate_expression_3d
        mni_img (nib.Nifti1Image): Loaded MNI image with affine
        resolution (int): Step size in voxel units (1 = full resolution)

    Returns:
        grid_values (np.ndarray): 3D array of interpolated expression values
        X, Y, Z (np.ndarray): 3D meshgrid arrays in MNI space (millimeters)
    """

    affine = mni_img.affine
    shape = mni_img.shape

    # Create grid of voxel indices
    i = np.arange(0, shape[0], resolution)
    j = np.arange(0, shape[1], resolution)
    k = np.arange(0, shape[2], resolution)

    I, J, K = np.meshgrid(i, j, k, indexing='ij')

    # Flatten and convert voxel indices to world (MNI) coordinates
    vox_coords = np.column_stack((I.ravel(), J.ravel(), K.ravel(), np.ones(I.size)))
    world_coords = vox_coords @ affine.T
    world_coords = world_coords[:, :3]  # Drop the homogeneous coordinate

    # Reshape to grid
    grid_shape = (len(i), len(j), len(k))

    # Also reshape world coordinate grids for optional visualization
    X = world_coords[:, 0].reshape(grid_shape)
    Y = world_coords[:, 1].reshape(grid_shape)
    Z = world_coords[:, 2].reshape(grid_shape)

    grid_points = np.column_stack([
        X.ravel(),  # shape (N,)
        Y.ravel(),
        Z.ravel()
    ])  # shape (N, 3)

    # Build KD-tree
    tree = cKDTree(coords)

    distances, indices = tree.query(grid_points, k=search_k)
    points_within_r = [sum(d < search_radius) for d in distances]
    min_distance = np.array(points_within_r).reshape(X.shape) 

    return min_distance
