
# import libraries
from matplotlib_surface_plotting import plot_surf
from gene_viz.utils import load_mesh_geometry
import numpy as np


def ras2coords(mri_img, ras=(0,0,0)):
   M = mri_img.affine[:3, :3]
   abc = mri_img.affine[:3, 3]
   
   i,j,k = ras
   x,y,z = M.dot([i, j, k]) + abc
   
   return x,y,z

def create_plane_surf(x1=0,x2=1,z1=0,z2=1,y=0):
    ''' Create a plane mesh composed of 4 vertex (v0,v1,v2,v3) and two faces (f1, f2)'''

    # define coordonates of the 4 vertices of the square plane
    v0 = x1,y,z1 # bottom left
    v1 = x2,y,z1 # bottom right
    v2 = x1,y,z2 # top left
    v3 = x2,y,z2 # top right

    vertices_plane = np.array([v0, v1, v2, v3])

    # create the faces (triangles) forming the square plane
    f1 = [0,1,2] 
    f2 = [1,3,2] 
    faces_plane = np.array([f1,f2])

    return vertices_plane, faces_plane

def plot_surf_and_plane(mesh, mesh_overlay=None, mri_img=None, slice_i=0, slice_axes=1, **kwargs):
    
    ''' Plots a 3D surface mesh with an optional overlay and adds a plane to the visualization.

    Parameters:
        mesh (dict): Dictionary containing the mesh data with keys:
            'coords': numpy.ndarray of shape (n_vertices, 3), representing the vertex coordinates.
            'faces': numpy.ndarray of shape (n_faces, 3), representing the triangular faces.
        mesh_overlay (numpy.ndarray, optional): Array of values to overlay on the mesh. If None, a random overlay is generated.
        mri_img (nibabel.nifti1.Nifti1Image, optional): MRI image object to derive the plane coordinates. If None, the plane is created based on the mesh bounds.
        slice_i (int, optional): Index of the slice to use for the plane. Default is 0. #TODO
        slice_axes (int, optional): Axis along which the slice is taken. Default is 1.
        **kwargs: Additional keyword arguments for customization of the plot.
    '''
    # get vertices and faces of the mesh
    mesh_vertices = mesh['coords']
    mesh_faces = mesh['faces']

    # create an overlay of the mesh
    if mesh_overlay is None:
        mesh_overlay = np.ones(len(mesh_vertices))
    
    # get the MRI coordinates from the RAS coordinates
    if not mri_img is None:
        shape = mri_img.shape
        x1,y,z1 = ras2coords(mri_img, ras=(0,slice_i,0))
        x2,y,z2 = ras2coords(mri_img, ras=(shape[0],slice_i,shape[2]))
    else:
        x1,_,z1 = mesh_vertices.min(axis=0)
        x2,_,z2 = mesh_vertices.max(axis=0)
        y = slice_i

    # create the plane mesh
    vertices_plane,faces_plane = create_plane_surf(x1=x1,x2=x2,z1=z1,z2=z2,y=y)

    # create the overlay for the square
    overlay_plane = np.array([1,1,1,1])

    # update plane vertices and faces to match the brain mesh
    vertices = np.vstack([mesh_vertices, vertices_plane])
    faces = np.vstack([mesh_faces, faces_plane+len(mesh_vertices)])
    overlay = np.hstack([mesh_overlay, overlay_plane*2])

    # create the transparency of the plane
    alpha = np.ones(len(vertices))
    alpha[-4::] = alpha[-4::]*0.1

    # plot
    plot_surf(vertices, faces, overlay, 
          alpha_colour=alpha,
          **kwargs
         )
    
def concatenate_meshes(mesh_files, f_explode, overlays=None):
    vertices = []
    faces = []
    overlay = []
    for i, mesh_file in enumerate(mesh_files): 
        # load the mesh
        mesh = load_mesh_geometry(mesh_file)
        if i==0:
            # add the vertices and faces of the first mesh
            vertices = mesh['coords'] + (mesh['coords'].mean(axis=0)) * f_explode
            faces = mesh['faces']
            if overlays is None:
                overlay = np.ones(len(mesh['coords']))*1
            else: 
                overlay = overlays[i]
        else:
            # add the vertices and faces of the other meshes
            faces = np.vstack([faces, mesh['faces']+len(vertices)])
            vertices = np.vstack([vertices,  mesh['coords']+ (mesh['coords'].mean(axis=0)) * f_explode])
            if overlays is None:
                overlay = np.hstack([overlay, np.ones(len(mesh['coords']))*(i+2)])
            else: 
                overlay = np.hstack([overlay, overlays[i]])
    return vertices, faces, overlay