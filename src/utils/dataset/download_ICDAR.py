import os
import shutil
import requests
from PIL import Image
from tqdm import tqdm
import config as C

def txt161(big_img_path):
    NEW_IMG = "P_Koln_I_27fr.jpg"
    
    #read big image
    im_big = Image.open(big_img_path)
    im_big = im_big.rotate(180)
    new_size=(3159,3171)
    im_big = im_big.resize(new_size)

    # crop image
    left = 2073
    top = 71
    right = 2073+640
    bottom = 71+654
    im_crop = im_big.crop((left, top, right, bottom))
    
    # Save Image
    im_crop.save(os.path.join(os.path.dirname(big_img_path), NEW_IMG))

os.path.dirname(C.ICDAR_IMAGES_PATH)

ICDAR_FOLDER = os.path.join(os.path.dirname(C.ICDAR_IMAGES_PATH))
ICDAR_CSV = os.path.join("data", "icdar_download.csv")

class image_downloader:
    def __init__(self) -> None:
        pass

    def download_image(self, srclink, dstpath):
        res = requests.get(srclink)

        with open(dstpath,'wb') as f:
            f.write(res.content)


if os.path.exists(ICDAR_FOLDER):
    shutil.rmtree(ICDAR_FOLDER)
ICDAR_FOLDER = os.path.join(ICDAR_FOLDER, "images", "homer2")
os.makedirs(ICDAR_FOLDER)

downloader = image_downloader()

with open(ICDAR_CSV, "r") as csv_file:
    for line in tqdm(csv_file.readlines()[1:]):
        img_name, subfolder,link = line.strip().split(",")

        if not os.path.exists(os.path.join(ICDAR_FOLDER, subfolder)):
            os.mkdir(os.path.join(ICDAR_FOLDER, subfolder))

        downloader.download_image(link, os.path.join(ICDAR_FOLDER, subfolder, img_name))

        if subfolder == "txt161":
            txt161(os.path.join(ICDAR_FOLDER, subfolder, img_name))
