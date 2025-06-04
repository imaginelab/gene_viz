import nibabel as nib
import numpy as np
from skimage import measure
import pyvista as pv
import pandas as pd
import os

def generate_mesh(data, label, label_name):
    """ Generate a mesh for a specific label in the segmentation data.
    """
    # Select corresponding voxels
    mask = (data == label)

    # if np.sum(mask) < 20:  # less than 100 voxels
    #    print(f"Skipping label {label} ({label_name}), too few voxels.")
    #   continue

    # Create mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(mask.astype(float), level=0.5)
    # print(f"Verts: {verts.shape}, Faces: {faces.shape}")

    # Vertices and faces are weirdly not in the right format
    # Need to reformat to: [n_points, i, j, k] for use with PyVista
    faces_vtk = np.hstack([
        np.full((faces.shape[0], 1), 3),  # 3 = number of vertices per face
        faces
    ])

    mesh = pv.PolyData(verts, faces_vtk)

    # Apply smoothing filter
    smoothed_mesh = mesh.smooth_taubin(n_iter=100, pass_band=0.1)

    return mesh, smoothed_mesh

plot_it=False

if not os.path.exists('../../../mesh-png'):
    os.makedirs('../../../mesh-png')

# Load segmentation
nii = nib.load("aparc+aseg.mni152.v2.nii")
data = nii.get_fdata()

# To self, this is how you see all label values: np.unique(data).astype(int)

# Read in look up table to be able to look up label names
lut_df = pd.read_csv(
    'FreeSurferColorLUT.txt',
    delim_whitespace=True,
    comment='#',
    header=None,
    names=["ID", "Name", "R", "G", "B", "A"]
)

# Loop over label
for label in np.unique(data).astype(int):
    if label == 0:
        continue
    # Find label name in LUT
    label_name = lut_df[lut_df["ID"] == label]["Name"].values[0]
    print('Running label number: ' + str(label) + '; Label name: ' + label_name)

    mesh, smoothed_mesh = generate_mesh(data, label, label_name)

    if plot_it==True:
        # Save mesh to file
        mesh.save("mesh-ply/" + label_name + "smoothed_mesh.ply")
        smoothed_mesh.save("mesh-ply/" + label_name + "smoothed_mesh_smoothed.ply")
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh)
        plotter.screenshot("mesh-png/" + label_name + "smoothed_mesh.png")
        plotter.close()

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(smoothed_mesh)
        plotter.screenshot("mesh-png/" + label_name + "smoothed_mesh_smoothed.png")
        plotter.close()


print('Done')
