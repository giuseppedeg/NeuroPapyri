import os
import shutil
import requests
from tqdm import tqdm
import zipfile
import config as C

ALPUBV2_FOLDER = os.path.join(C.ALPUBV2_IMAGES_PATH)


if os.path.exists(ALPUBV2_FOLDER):
    shutil.rmtree(ALPUBV2_FOLDER)
os.makedirs(ALPUBV2_FOLDER)

with open(os.path.join(ALPUBV2_FOLDER, "datazip"), "wb") as f:
    response = requests.get("https://data.cs.mtsu.edu/al-pub/alpub_v2.zip", stream=True)
    total_length = response.headers.get('content-length')

    if total_length is None: # no content length header
        f.write(response.content)
    else:
        dl = 0
        total_length = int(total_length)
        for data in tqdm(response.iter_content(chunk_size=4096), desc="Downloading "):
            dl += len(data)
            f.write(data)
            done = int(50 * dl / total_length)

print("Extracting...\n")
with zipfile.ZipFile(os.path.join(ALPUBV2_FOLDER, "datazip"), 'r') as zip_ref:
    zip_ref.extractall(ALPUBV2_FOLDER)

# with zipfile.ZipFile(os.path.join(ALPUBV2_FOLDER, "data.zip"), 'r') as zip_ref:
#     for member in tqdm(zip_ref.infolist(), desc='Extracting '):
#         try:
#             zip_ref.extractall(ALPUBV2_FOLDER)
#         except zipfile.error as e:
#             pass
    
os.remove(os.path.join(ALPUBV2_FOLDER, "datazip"))