from gene_viz.interpolation.cross_validation import cross_validate
import os
import pandas as pd
import numpy as np
from gene_viz.utils import get_michack_data_path,get_data_path

#turn this into a class tha loads the data once on initialisation
#and then returns coordinates and samples for a given gene.




def load_data(gene_name='PVALB',flip_lr=True):
    """
    Load the Michack project data, including expression and coordinates.
    
    Parameters
    ----------
    flip_lr : bool, optional
        If True, flips the left-right coordinates. Default is True.
    
    Returns
    -------
    expression : pd.DataFrame
        Gene expression data indexed by well ID.
    coords : pd.DataFrame
        Coordinates data indexed by well ID.
    """
    data_path = get_michack_data_path()
    
    # Load cached expression and coordinates
    expression_file = os.path.join(data_path, 'point_expression_data.csv')
    coords_file = os.path.join(data_path, 'coords_data.csv')
    
    if not os.path.exists(expression_file) or not os.path.exists(coords_file):
        raise FileNotFoundError("Cached data files not found. Please download the data first.")
    
    expression = pd.read_csv(expression_file, index_col=0)
    coords = pd.read_csv(coords_file, index_col=0)
    sample_coords = np.array(coords)
    samples = np.array(expression[gene_name]).ravel()


    if flip_lr:
        #LR flipping and stacked
        flipped_coords = np.copy(sample_coords)
        flipped_coords[:, 0] = -flipped_coords[:, 0]
        flipped_samples = np.copy(samples)
        #stack
        stacked_coords = np.vstack((sample_coords, flipped_coords))
        stacked_samples = np.hstack((samples, flipped_samples))
        sample_coords = stacked_coords
        samples = stacked_samples
    
    return sample_coords, samples


