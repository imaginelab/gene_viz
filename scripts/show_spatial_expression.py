
# JACK HIGHTON, KING'S COLLEGE LONDON, 2025

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import abagen
from nilearn.datasets import MNI152_FILE_PATH

directory1 = 'michack_project_data'
os.makedirs(directory1, exist_ok=True)

# File paths
mni_output_path = os.path.join(directory1, 'MNI152_T1_1mm.nii.gz')
gene_name = 'SCN2A'
gene_output_path = os.path.join(directory1, f'{gene_name}_spatial_expression.nii.gz')

# Load or save MNI atlas
if os.path.exists(mni_output_path):
    print(f"Loading MNI atlas from {mni_output_path}")
    mni_img = nb.load(mni_output_path)
else:
    print(f"Saving MNI atlas to {mni_output_path}")
    mni_img = nb.load(MNI152_FILE_PATH)
    nb.save(mni_img, mni_output_path)

mni_vol = np.array(mni_img.dataobj)

# Load or compute gene expression map
if os.path.exists(gene_output_path):
    print(f"Loading {gene_name} expression from {gene_output_path}")
    gene_img = nb.load(gene_output_path)
    gene_vol = np.array(gene_img.dataobj)
    vol = {gene_name: gene_vol}
else:
    print(f"Downloading and saving expression for {gene_name}")
    vol = abagen.get_interpolated_map(
        [gene_name], mask=mni_img, lr_mirror='bidirectional',
        reannotated=False, ibf_threshold=0.1
    )
    gene_img = nb.Nifti1Image(vol[gene_name], affine=mni_img.affine)
    nb.save(gene_img, gene_output_path)

# Plot function
def plot_volumetrics(gene_name, mni_volume, gene_vols, sections=[80, 100, 60]):
    fig, axes = plt.subplots(3, 3, figsize=(30, 30))
    axes[0, 1].set_title(f'Spatial bulk expression: {gene_name}', fontsize=80)

    for a in np.arange(3):
        mni_section = np.flipud(mni_volume.take(sections[a], axis=a).T)
        gene_im = np.flipud(gene_vols[gene_name].take(sections[a], axis=a).T)
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

# Plot
fig = plot_volumetrics(gene_name, mni_vol, vol, sections=[80, 100, 60])
plt.show()