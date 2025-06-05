import nibabel as nib
import numpy as np
from skimage import measure
import pyvista as pv
import pandas as pd
import os
import sys
from gene_viz.utils import save_mesh_geometry, get_data_path, ras_array2coords

def generate_mesh(data, label, label_name):
    """ Generate a mesh for a specific label in the segmentation data.
    """
    # Select corresponding voxels
    mask = (data == label)

    # Consider using this if you want to skip small labels
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

# Can use this to create png screenshots of the meshes
plot_it=False
data_path = get_data_path()
mesh_png_path = os.path.join(data_path,'..', 'mesh-png')
if not os.path.exists(os.path.join(data_path,'..','mesh-png')) and plot_it==True:
    os.makedirs(os.path.join(data_path,'..','mesh-png'))

# If you have run the download script, the data folder will already exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Load segmentation
gen_mesh_path = os.path.join(data_path,'..', 'gene_viz', 'generate-meshes')
nii = nib.load(os.path.join(gen_mesh_path,"aparc+aseg.mni152.v2.nii"))
data = nii.get_fdata()

# Read in look up table to be able to look up label names
lut_df = pd.read_csv(os.path.join(gen_mesh_path, 'FreeSurferColorLUT.txt'),
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

    # Generate mesh
    mesh, smoothed_mesh = generate_mesh(data, label, label_name)
    coords =ras_array2coords(nii,np.array(smoothed_mesh.points))

    # Create corrds/face dictionary
    surf_dict = {
        'coords': coords,  # numpy array of points (N,3)
        'faces': np.array(smoothed_mesh.faces.reshape(-1, 4)[:, 1:] ) # skip the count number before each face
    }

    # Save mesh to file
    outpath=os.path.join(data_path, label_name + '_meshfile.surf.gii')
    save_mesh_geometry(outpath, surf_dict)

    if plot_it==True:
        # Save mesh to file
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh)
        plotter.screenshot(os.path.join(mesh_png_path,  label_name + "smoothed_mesh.png"))
        plotter.close()

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(smoothed_mesh)
        plotter.screenshot(os.path.join(mesh_png_path,  label_name + "smoothed_mesh_smoothed.png"))
        plotter.close()


print('Done')
