from gene_viz.interpolation.cross_validation import cross_validate
import os
import numpy as np
from gene_viz.utils import get_michack_data_path, get_data_path
import nibabel as nb

def mni_atlas_loader():

    mni_output_path = os.path.join(get_michack_data_path(), 'MNI152_T1_1mm.nii.gz')
    mni_img = nb.load(mni_output_path)
    mni_vol = np.array(mni_img.dataobj)
    affine = mni_img.affine

    return mni_img, mni_vol, affine