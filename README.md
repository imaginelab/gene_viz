# gene_viz
MIC HACK project repo


https://github.com/kwagstyl/matplotlib_surface_plotting/tree/main

## Create environment 

```bash
conda env create -f environment.yml
conda activate gene_viz
```


### Add matplotlib_surface_plotting package 

Git clone matplotlib_surface_plotting from mathrip github
```bash
git clone https://github.com/mathrip/matplotlib_surface_plotting.git
```
And install in environment
```bash
pip install -e . 
```

### Getting started

## Download the gene expression data

## Get the cortical and subcortical mesh files

1. Download the FSLR cortical mesh by running `python gene_viz/downloaders/download_cortical_meshes.py`. This will generate a folder `gene_viz/data`, and download the FSLR `fs_LR.32k.<hemisphere>.pial.surf.gii` mesh files into it.
2. Generate the regional mesh files for the APARC and ASEG regions in MNI format by running `python gene_viz/generate-meshes/create-surface-mesh.py`. This will create a single mesh file `<region>_meshfile.ply` for each region and save them in `gene_viz/data`. 

As a side note: Currently we are using fs LR surface meshes, they are a bit too smooth but they'll do for now. In the future, we would like to generate more tailored MNI152 meshes fit the MNI anatomy better. Further, the APARC and ASEG region meshes have still have some holes, likely due to the thin cortical band of the template we used. Future work should fix this.

## Example images
expression_point_density (red is 10 points within 10mm):
![expression_point_density (red is 10 points within 10mm)](https://github.com/user-attachments/assets/6ab0f5ad-e7a1-4649-8df7-107cd4eb4320)
SCN2A expression ith alpha based on point density:
![SCN2A_spatial_expression_alpha_point_density](https://github.com/user-attachments/assets/5c940d08-ab97-4a12-8fdc-a3cf7fb5107e)

## Example notebooks

Main notebook to plot gene expression : [main_plotting](/gene_viz/notebooks/main_plotting.ipynb)

Notebook to plot the mesh and a plane : [plot_plane_surface](/gene_viz/notebooks/plot_plane_surface.ipynb)

