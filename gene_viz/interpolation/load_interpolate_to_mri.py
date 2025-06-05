# JACK HIGHTON, KING'S COLLEGE LONDON, 2025

import os
import numpy as np
import matplotlib.pyplot as plt

#import abagen
#from abagen import datasets, samples_, io
#from abagen.utils import flatten_dict
import nibabel as nb

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


def interpolate_expression_3d(coords_df, expression_series):
    """
    Interpolates gene expression values in 3D space.

    Parameters:
        coords_df (pd.DataFrame): DataFrame with index as 'well_id' and columns ['x', 'y', 'z']
        expression_series (pd.Series): Series with index as 'well_id' and gene expression values

    Returns:
        interp_func (callable): Function that accepts (x, y, z) coordinates and returns interpolated expression
    """
    # Join data on well_id
    df = coords_df.join(expression_series.rename("expression"), how='inner')
    # Extract arrays for interpolation
    points = df[['x', 'y', 'z']].values
    values = df['expression'].values
    # Create interpolator
    interp_func = LinearNDInterpolator(points, values)

    return interp_func

def make_3d_interpolated_grid_mni(interp_func, mni_img, resolution=1):
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
    import numpy as np

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

    # Interpolate at world coordinates
    values = interp_func(world_coords)

    # Reshape to grid
    grid_shape = (len(i), len(j), len(k))
    grid_values = values.reshape(grid_shape)

    # Also reshape world coordinate grids for optional visualization
    X = world_coords[:, 0].reshape(grid_shape)
    Y = world_coords[:, 1].reshape(grid_shape)
    Z = world_coords[:, 2].reshape(grid_shape)

    return grid_values, X, Y, Z

# Plot function
def plot_volumetrics(gene_name, mni_volume, gene_vols, sections=[80, 100, 60]):
    fig, axes = plt.subplots(3, 3, figsize=(30, 30))
    axes[0, 1].set_title(f'Spatial bulk expression: {gene_name}', fontsize=80)

    for a in np.arange(3):
        mni_section = np.flipud(mni_volume.take(sections[a], axis=a).T)
        gene_im = np.flipud(gene_vols.take(sections[a], axis=a).T)
        gene_im_mask = np.ma.masked_array(gene_im, gene_im == 0)

        axes[a, 0].imshow(mni_section, cmap='Greys_r', vmin=100, vmax=255)
        axes[a, 0].axis('off')

        axes[a, 1].imshow(mni_section, cmap='Greys_r')
        axes[a, 1].imshow(gene_im_mask, cmap='turbo')
        axes[a, 1].axis('off')

        axes[a, 2].imshow(mni_section, cmap='Greys_r', vmin=100, vmax=255)
        axes[a, 2].imshow(gene_im_mask, cmap='turbo', alpha=0.5)
        axes[a, 2].axis('off')

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

def plot_volumetrics_alpha(gene_name, mni_volume, gene_vols, alpha_vol, sections=[80, 100, 60]):
    fig, axes = plt.subplots(3, 3, figsize=(30, 30))
    axes[0, 1].set_title(f'Spatial bulk expression: {gene_name}', fontsize=80)

    for a in np.arange(3):
        # Base anatomical slice
        mni_section = np.flipud(mni_volume.take(sections[a], axis=a).T)

        # Gene expression slice
        gene_im = np.flipud(gene_vols.take(sections[a], axis=a).T)
        gene_im_mask = np.ma.masked_array(gene_im, gene_im == 0)

        # Alpha map slice
        alpha_slice = np.flipud(alpha_vol.take(sections[a], axis=a).T)
        alpha_masked = np.ma.masked_array(alpha_slice, gene_im == 0)

        axes[a, 0].imshow(mni_section, cmap='Greys_r', vmin=100, vmax=255)
        axes[a, 0].axis('off')

        # Middle: gene expression over anatomy (full alpha)
        axes[a, 1].imshow(mni_section, cmap='Greys_r')
        axes[a, 1].imshow(gene_im_mask, cmap='turbo')
        axes[a, 1].axis('off')

        # Right: gene expression over anatomy with alpha map
        axes[a, 2].imshow(mni_section, cmap='Greys_r', vmin=100, vmax=255)
        axes[a, 2].imshow(gene_im_mask, cmap='turbo', alpha=alpha_masked)
        axes[a, 2].axis('off')

    fig.tight_layout()
    return fig