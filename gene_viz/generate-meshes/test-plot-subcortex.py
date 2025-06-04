import os, sys
#os.chdir('/Users/lenadorfschmidt/Documents/KCL/MIC-HACK2025/gene_viz/gene_viz/generate-meshes')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib_surface_plotting.matplotlib_surface_plotting import plot_surf
import nibabel as nb
import numpy as np

# parent_dir = '/Users/lenadorfschmidt/Documents/KCL/MIC-HACK2025/gene_viz/gene_viz'
# sys.path.append(parent_dir)
# Add the parent directory so we can use utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import read_ply
import random

vertices, faces = read_ply('mesh-ply/Right-Cerebellum-Cortexmeshfile.ply')
overlay = np.array([random.random() for _ in range(len(faces))])
plot_surf( vertices, faces, overlay,filename='test-right-cerebellum.png',rotate=[0,45,90,120,180])

vertices, faces = read_ply('mesh-ply/Right-Hippocampusmeshfile.ply')
overlay = np.array([random.random() for _ in range(len(faces))])
plot_surf( vertices, faces, overlay,filename='test-right-hippocampus.png',rotate=0)
