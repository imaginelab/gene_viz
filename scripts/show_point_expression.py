
# JACK HIGHTON, KING'S COLLEGE LONDON, 2025

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import abagen
from abagen import datasets, samples_, io
from abagen.utils import flatten_dict

# --- Set up paths ---
directory1 = 'michack_project_data'
os.makedirs(directory1, exist_ok=True)

# Cached CSV file paths
expression_file = os.path.join(directory1, 'point_expression_data.csv')
coords_file = os.path.join(directory1, 'coords_data.csv')

# Check if raw microarray donor data is already downloaded
donor_dirs = [f for f in os.listdir(directory1) if f.startswith('point_')]
need_fetch = len(donor_dirs)<0 

# --- Fetch microarray data if missing ---
if need_fetch:
    print("Downloading donor data to directory1...")
    files = datasets.fetch_microarray(data_dir=directory1, donors='all', verbose=1, n_proc=1)
else:
    print("Donor data already exists in directory1.")
    files = datasets.fetch_microarray(data_dir=directory1, donors='all', verbose=0, n_proc=1)

# --- Load or compute expression and coordinates ---
if os.path.exists(expression_file) and os.path.exists(coords_file):
    print("Loading cached expression and coordinates...")
    expression = pd.read_csv(expression_file, index_col=0)
    coords = pd.read_csv(coords_file, index_col=0)
else:
    print("Computing expression and coordinates from donor data...")
    expression, coords = abagen.get_samples_in_mask(mask=None)
    expression.to_csv(expression_file)
    coords.to_csv(coords_file)

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

plt.figure(figsize=(10, 6))
plt.scatter(coords['y'], coords['z'], c=single_gene, cmap='turbo')
plt.colorbar(label=f"{gene_name} expression")
plt.title(f"Spatial expression of {gene_name}")
plt.xlabel("Y (MNI)")
plt.ylabel("Z (MNI)")
plt.tight_layout()
plt.show()