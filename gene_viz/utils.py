#create get_data_path function to return the path to the data directory
#data path is the root/data directory

# gene_viz/utils.py

import os
def get_data_path():
    """
    Returns the path to the data directory.
    The data directory is expected to be at the root level of the gene_viz package.
    """
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the root directory of the gene_viz package
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Define the data directory
    data_path = os.path.join(root_dir, 'data')
    
    return data_path