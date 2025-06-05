# Load gene expression data - Konrad

from gene_viz.data_loader import load_data

gene_name = 'PVALB'

coords, samples = load_data(gene_name)

# Select the gene - Konrad
#done

# Specify gene name - Konrad
#done

# Load in meshes - Konrad, Lena

from gene_viz.utils import get_data_path, load_mesh_geometry
import os
cortical_mesh_file_path = os.path.join(get_data_path(),'fs_LR.32k.L.pial.surf.gii')
mesh = load_mesh_geometry(cortical_mesh_file_path)

# Interpolate gene data to meshes - Konrad, Lena

from gene_viz.interpolation.interpolation_core import interpolate
interpolated_values = interpolate(samples,coords, mesh['coords'])

print(interpolated_values)
# Load in MRI - Jack
#done

# Interpolate gene data to MRI slice - Jack

import matplotlib.pyplot as plt
from gene_viz.load_interpolate_to_mri import plot_volumetrics, plot_volumetrics_alpha

fig = plot_volumetrics_alpha(gene_name, mni_vol, grid_values, alpha_mask, sections=[80, 100, 60])

fig = plot_volumetrics(gene_name, mni_vol, grid_values, sections=[80, 100, 60])
plt.show()

fig = plot_volumetrics(gene_name, mni_vol, min_distance_vol, sections=[80, 100, 60])
plt.show()

alpha_mask = (min_distance_vol-np.min(min_distance_vol)) / np.max(min_distance_vol)
fig = plot_volumetrics_alpha(gene_name, mni_vol, grid_values, alpha_mask, sections=[80, 100, 60])
plt.show()

# Visualise data in meshes - Mathilde
# Visualise data in slice of MRI - Jack


