import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib_surface_plotting.matplotlib_surface_plotting import plot_surf
import nibabel as nb
import numpy as np
from gene_viz.utils import read_ply
import random

# Load in surface mesh
vertices, faces = read_ply('../../data/Right-Cerebellum-Cortex_meshfile.ply')
# Creating random number - replace this with real data
overlay = np.array([random.random() for _ in range(len(faces))])
# Plot mesh with random overlay
plot_surf( vertices, faces, overlay,filename='test-right-cerebellum.png',rotate=[0,45,90,120,180])

vertices, faces = read_ply('../../data/Right-Hippocampus_meshfile.ply')
overlay = np.array([random.random() for _ in range(len(faces))])
plot_surf( vertices, faces, overlay,filename='test-right-hippocampus.png',rotate=0)
