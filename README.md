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

#currently we are using fs LR surface meshes. A bit too smooth but they'll do for now.
#it's possible some MNI152 tailored meshes fit the MNI anatomy better, but not dealing with that now.

## Example images
expression_point_density (red is 10 points within 10mm):
![expression_point_density (red is 10 points within 10mm)](https://github.com/user-attachments/assets/6ab0f5ad-e7a1-4649-8df7-107cd4eb4320)
