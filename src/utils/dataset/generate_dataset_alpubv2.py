import random
import functools
import os
from PIL import Image
from glob import glob
import csv
import shutil
from tqdm import tqdm
import config as C
from utils.utils import targets_dico, targets as trgs

n_train = 32768
n_test = 1024
k = 10

path = C.ALPUBV2_DATASET_PATH
base_paths = glob(os.path.join(os.path.join(C.ALPUBV2_IMAGES_PATH, "images"), '*'))

random.seed(C.SEED_01)

if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)
os.makedirs(os.path.join(path, 'train'))
os.makedirs(os.path.join(path, 'test'))

letter_names = {targets_dico[os.path.basename(path).replace('LunateSigma', 'Sigma').replace('Chi', 'Khi')]: path for path in base_paths}

letters_paths = {name: glob(os.path.join(path, '*')) for name, path in letter_names.items()}

test_paths = []

f = open(os.path.join(path,'test/datas.csv'), 'w')
writer = csv.writer(f)
writer.writerow(['filename', 'targets', 'transcription'])


for i in tqdm(range(n_test), desc="Test "):
    targets = []
    images = []
    for j in range(k):
        letter, paths = random.choice(list(letters_paths.items()))
        targets.append(letter)
        img_path = random.choice(paths)
        img = Image.open(img_path)
        images.append(img)
        test_paths.append(img_path)

    total_width = functools.reduce(lambda a, b: a + b.size[0], images, 0)
    max_height = functools.reduce(lambda a, b: a if a > b.size[1] else b.size[1], images, 0)
    img = Image.new('RGB', (total_width, max_height))
    x = 0
    for l_img in images:
        w, h = l_img.size
        img.paste(l_img, (x, max_height // 2 - h//2))
        x += w
    targets_set = list(set(targets))
    trg = [1 if trg in targets_set else 0 for trg in trgs]
    
    writer.writerow([f'{i}.jpg', trg, targets])
    img.save(os.path.join(path, f'test/{i}.jpg'), quality=100)
f.close()

f = open(os.path.join(path, 'train/datas.csv'), 'w')
writer = csv.writer(f)
writer.writerow(['filename', 'targets', 'transcriptions'])

for i in tqdm(range(n_train), desc="Train "):
    targets = []
    images = []
    for j in range(k):
        letter, paths = random.choice(list(letters_paths.items()))
        targets.append(letter)
        img_path = random.choice(paths)
        while img_path in test_paths:
            img_path = random.choice(paths)
        img = Image.open(img_path)
        images.append(img)

    total_width = functools.reduce(lambda a, b: a + b.size[0], images, 0)
    max_height = functools.reduce(lambda a, b: a if a > b.size[1] else b.size[1], images, 0)
    img = Image.new('RGB', (total_width, max_height))
    x = 0
    for l_img in images:
        w, h = l_img.size
        img.paste(l_img, (x, max_height // 2 - h//2))
        x += w
    targets_set = list(set(targets))
    trg = [1 if trg in targets_set else 0 for trg in trgs]
    
    writer.writerow([f'{i}.jpg', trg, targets])
    img.save(os.path.join(path, f'train/{i}.jpg'), quality=100)
f.close()
