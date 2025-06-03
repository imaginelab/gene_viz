import os
import requests
from tqdm import tqdm
from gene_viz.utils import get_data_path

def download_cortical_meshes(hemisphere='L'):
    #left and right pial surface meshes for fs_LR 32k
    
    url = "https://github.com/DiedrichsenLab/fs_LR_32/raw/refs/heads/master/fs_LR.32k.{}.pial.surf.gii"
    url = url.format(hemisphere.upper())
    filename = "fs_LR.32k.{}.pial.surf.gii"
    filename = filename.format(hemisphere.upper())
    data_path = get_data_path()
    file_path = os.path.join(data_path, filename)

    if os.path.exists(file_path):
        print(f"{filename} already exists at {file_path}. Skipping download.")
        return

    os.makedirs(data_path, exist_ok=True)
    print(f"Downloading {filename}...")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(file_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

        print(f"{filename} downloaded successfully to {file_path}.")
    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")
if __name__ == "__main__":
    for hemisphere in ['L', 'R']:
        download_cortical_meshes(hemisphere)
