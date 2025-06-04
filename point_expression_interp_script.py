
# JACK HIGHTON, KING'S COLLEGE LONDON, 2025

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import abagen
from abagen import datasets, samples_, io
from abagen.utils import flatten_dict
import nibabel as nb

from scipy.interpolate import LinearNDInterpolator

# --- Set up paths ---
directory1 = 'michack_project_data'
os.makedirs(directory1, exist_ok=True)

# Cached CSV file paths
expression_file = os.path.join(directory1, 'point_expression_data.csv')
coords_file = os.path.join(directory1, 'coords_data.csv')

# --- Fetch microarray data if missing ---
print("Donor data already exists in directory1.")
files = datasets.fetch_microarray(data_dir=directory1, donors='all', verbose=0, n_proc=1)

# --- Load or compute expression and coordinates ---
print("Loading cached expression and coordinates...")
expression = pd.read_csv(expression_file, index_col=0)
coords = pd.read_csv(coords_file, index_col=0)

# --- Annotate structures ---
for donor, data in files.items():
    annot = data['annotation']
    ontology = io.read_ontology(data['ontology']).set_index('id')
    annot = samples_.update_mni_coords(annot)
    sid = np.asarray(annot['structure_id'])
    structure = np.asarray(ontology.loc[sid, 'structure_id_path']
                           .apply(samples_._get_struct))
    annot = annot.assign(structure=structure)
    files[donor]['annotation'] = annot

cols = ['well_id', 'structure_name', 'structure']
structure_names = np.asarray(pd.concat(flatten_dict(files, 'annotation'))[cols], dtype=str)
well_id, structure_names = np.asarray(structure_names[:, 0], 'int'), structure_names[:, 1:]
structure_names = pd.DataFrame(structure_names, columns=['structure_name', 'structure'], index=well_id)

print(np.unique(structure_names['structure']))

# --- Visualize gene expression ---
gene_name = 'PVALB'
if gene_name not in expression.columns:
    raise ValueError(f"Gene '{gene_name}' not found in expression data.")

single_gene = expression[gene_name]

#plt.figure(figsize=(10, 6))
#plt.scatter(coords['y'], coords['z'], c=single_gene, cmap='turbo')
#plt.colorbar(label=f"{gene_name} expression")
#plt.title(f"Spatial expression of {gene_name}")
#plt.xlabel("Y (MNI)")
#plt.ylabel("Z (MNI)")
#plt.tight_layout()
#plt.show()

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

def make_3d_interpolated_grid(interp_func, 
                              x_range, y_range, z_range, 
                              resolution=50):
    """
    Generate a 3D grid of interpolated gene expression values.

    Parameters:
        interp_func (callable): Interpolator function from interpolate_expression_3d
        x_range, y_range, z_range (tuple): Each is (min, max) for the respective axis
        resolution (int): Number of points along each axis

    Returns:
        grid_values (np.ndarray): 3D array of interpolated expression values
        X, Y, Z (np.ndarray): 3D meshgrid arrays for coordinates
    """
    # Create 1D axes
    x = np.linspace(*x_range, resolution*int(x_range[1]-x_range[0]))
    y = np.linspace(*y_range, resolution*int(y_range[1]-y_range[0]))
    z = np.linspace(*z_range, resolution*int(z_range[1]-z_range[0]))

    # Create 3D grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten the grid for interpolation
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # Interpolate
    values = interp_func(points)

    # Reshape back to 3D
    grid_values = values.reshape((resolution*int(x_range[1]-x_range[0]), resolution*int(y_range[1]-y_range[0]), 
                                  resolution*int(z_range[1]-z_range[0])))

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

interp = interpolate_expression_3d(coords, single_gene)

# Define bounding box from your coordinates
#x_min, x_max = coords['x'].min(), coords['x'].max()
#y_min, y_max = coords['y'].min(), coords['y'].max()
#z_min, z_max = coords['z'].min(), coords['z'].max()

# Define bounding box from your coordinates
# Load MNI atlas
mni_output_path = os.path.join(directory1, 'MNI152_T1_1mm.nii.gz')
print(f"Loading MNI atlas from {mni_output_path}")
mni_img = nb.load(mni_output_path)
mni_vol = np.array(mni_img.dataobj)

x_min, x_max = np.ceil(-mni_vol.shape[0]/2), np.ceil(mni_vol.shape[0]/2)
y_min, y_max = np.ceil(-mni_vol.shape[1]/2), np.ceil(mni_vol.shape[1]/2)
z_min, z_max = np.ceil(-mni_vol.shape[2]/2), np.ceil(mni_vol.shape[2]/2)

# Create the grid
grid_values, X, Y, Z = make_3d_interpolated_grid(
    interp_func=interp,
    x_range=(x_min, x_max),
    y_range=(y_min, y_max),
    z_range=(z_min, z_max),
    resolution=1
)

# Plot
fig = plot_volumetrics(gene_name, mni_vol, grid_values, sections=[80, 100, 60])
plt.show()

chk=1