#!/bin/bash

mri_label2vol --reg $FREESURFER_HOME/average/mni152.register.dat --seg $SUBJECTS_DIR/fsaverage/mri/aparc+aseg.mgz --temp $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz --o aparc+aseg.mni152.v2.mgz

mri_convert aparc+aseg.mni152.v2.mgz aparc+aseg.mni152.v2.nii


