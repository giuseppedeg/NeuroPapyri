import os
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont
import csv
import shutil
import json
from statistics import mean
from utils.utils import targets_dico, targets_maj, targets as targets_min
from tqdm import tqdm
import config as C

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_TTL = 100
MIN_CHAR = 5
MAX_CHAR = 10

SAVE_FULL_IMAGE = False

np.random.seed(seed=C.SEED_01)

images_path = C.ICDAR_IMAGES_PATH
cocojson_path = C.ICDAR_COCOJSON_PATH
output_lines = C.ICDAR_DATASET01_PATH


debug_bbox_color_dict = {
    "θ": "bisque",
    "α": "black",
    "β": "red",
    "τ": "cadetblue",
    "ξ": "coral",
    "ε": "darkcyan",
    "κ": "darksalmon",
    "ω": "forestgreen",
    "μ": "khaki",
    "φ": "lime",
    "ρ": "olivedrab",
    "π": "palevioletred",
    "γ": "pink",
    "ν": "salmon",
    "λ": "sandybrown",
    "η": "blue",
    "υ": "cyan",
    "ψ": "darkolivegreen",
    "χ": "darkturquoise",
    "δ": "fuchsia",
    "ο": "lavenderblush",
    "ι": "lightseagreen",
    "ϲ": "mediumpurple",
    "ζ": "orange"
}


def isinbbox(coord, a) -> bool:
    '''
    Return true if coords are in the bbox's annotation.
    
    Args:
        coord (tuple): Coords used to detect if they are in the bbox.
        a (dict): A dict representing a annotation (Coco json format).
    '''
    x, y = coord[0], coord[1]
    bbox = a['bbox']
    return (bbox[0] <= x and x <= bbox[0] + bbox[2]) and (bbox[1] <= y and y <= bbox[1] + bbox[3])

def centeran(a):
    '''
    Return the center of an annotation.
    
    Args:
        a (dict): A dict representing a annotation (Coco json format).
    '''
    bbox = a['bbox']
    return bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2

def save_line(coord1, coord2, image, idx, dist='train'):
    '''
    Save an image of a line from coords and an image.
    
    Args:
        coord1 (tuple): Coords of the top left rectangle of the line.
        coord2 (tuple): Coords of the bottom right rectangle of the line.
        image (PIL.Image): A PIL image  used to crop the line.
        idx (int): index of the line, use for the file name.
        dist (str): name of the folder where the image is saved.
    '''
    i = image.crop((*coord1, *coord2))
    i.save(os.path.join(output_lines, f'{dist}/{idx}.jpg'))

def save_incsv(idx, line, img_path, fcsv):
    '''
    Save the line in CSV.
    
    Args:
        idx (int): index of the line, use for the file name.
        line (list): an array representing the transcription of the line.
        fcsv (): the writer to write a line in the csv.
    '''
    line_targets = [c['letter'] for c in line]
    
    targets_set = list(set(line_targets))

    if line_targets[0] in targets_min:
        trgs = targets_min
    else:
        trgs = targets_maj

    trg = [1 if trg in targets_set else 0 for trg in trgs]
    
    fcsv.writerow([f'{idx}.jpg', trg, line_targets, len(line), img_path])


if os.path.exists(output_lines):
    shutil.rmtree(output_lines)
os.makedirs(os.path.join(output_lines, "train"))
os.mkdir(os.path.join(output_lines, "test"))

with open(cocojson_path) as f:
    cocojson = json.load(f)

images = []
for im in cocojson['images']:
        lst = im['img_url'].split('/')
        im['img_path'] = os.path.join(*lst[-3:])
        im['annotations'] = []
        for an in cocojson['annotations']:
            if an['image_id'] == im['id']:
                for c in cocojson['categories']:
                    if an['category_id'] == c['id']:
                        an['letter'] = c['name'].replace('ϲ', 'σ')
                        break
                im['annotations'].append(an)
        images.append(im)

# Creation of CSV writer
ftrain = open(os.path.join(output_lines, 'train/datas.csv'), 'w')
train_writer = csv.writer(ftrain)
train_writer.writerow(['filename', 'targets', 'transcription', 'len', 'image_path'])

ftest = open(os.path.join(output_lines, 'test/datas.csv'), 'w')
test_writer = csv.writer(ftest)
test_writer.writerow(['filename', 'targets', 'transcription', 'len', 'image_path'])

total_lines = np.array([])
idx_line = 0
idx_train = 0
idx_test = 0
# Iterate through images.
for idx, im in tqdm(enumerate(images), desc="Images"):
    #print(f'{idx}/{len(images)}')
    img_path = os.path.join(*im['img_url'].split('/')[-3:])
    # Try to open the file, if it failed it's due to a bad encoding file name, so we fix it here.
    # It's specific for our datas, maybe it isnt necessary for other datas
    try:
        image = Image.open(os.path.join(images_path, im['img_path'])).convert('RGB')
    except FileNotFoundError:
        try:
            image = Image.open(os.path.join(images_path, im['img_path']).replace('ö', 'o')).convert('RGB')
        except FileNotFoundError:
            try:
                image = Image.open(os.path.join(images_path, im['img_path']).replace('ö', 'ö')).convert('RGB')
            except FileNotFoundError:
                image = Image.open(os.path.join(images_path, im['img_path']).replace('homer32', 'homer2')).convert('RGB')

    
    image_clean = image.copy()
    draw_img = ImageDraw.Draw(image)
    unicode_font = ImageFont.truetype("DejaVuSans.ttf", 24)

    an = np.array([])
    for a in im['annotations']:
        an = np.append(an, a.copy())
    # Sort the annotations list using the x coordinate.
    an = sorted(an, key=lambda a: a['bbox'][0])

    curr, an = an[0], an[1:]
    vx, vy = 1, 0
    # place cursor in the center of the annotation.
    x, y = centeran(curr)
    px, py = x, y
    lines = [[curr]]
    traces = [[]]
    i_ligne = 0
    ttl = MAX_TTL
    vxs = [vx]
    vys = [vy]
    # While there is annotations in the list.
    while len(an) > 0:
        # Iterate over annotations.
        for i in range(len(an)):
            # if the cursor is in the annotation
            if isinbbox((x, y), an[i]) and curr['bbox'] != an[i]['bbox']:
                # Get the annotation and delete it from the list. It becomes the current one.
                prev = curr
                curr = an[i]
                an = np.delete(an, i)
                
                px, py = centeran(prev)
                traces[i_ligne].append([(px, py), (x, y)])
                x, y = centeran(curr)
                
                # Compute moving average 
                if len(vxs) > 2:
                    vxs = [*vxs[1:], x-px]
                    vys = [*vys[1:], y-py]
                else:
                    vxs = [*vxs, x-px]
                    vys = [*vys, y-py]
                
                vx = mean(vxs)
                vy = mean(vys)
                
                # Debug
                if vx == 0:
                    print(vxs)
                    print(prev)
                    print(curr)
                
                # Normalize vx and vy.
                vy /= vx
                vx /= vx
                
                # Add the line in the list for this image and reset TTL.
                lines[i_ligne].append(curr)
                ttl=MAX_TTL
                break
        # Moves the cursor
        x+=vx
        y+=vy
        # Decrease TTL.
        ttl-=(abs(vx)+abs(vy))
        # if cursor is out of the image.
        if x > im['width'] or y > im['height'] or ttl <= 0:# or len(lines[i_ligne]) >= MAX_CHAR:
            ttl = MAX_TTL
            curr, an = an[0], an[1:]
            lines.append([curr])
            x, y = centeran(curr)
            i_ligne += 1
            px, py = x, y
            traces.append([])
    # Sort line using the y coord of the first annotation.
    lines = sorted(lines, key=lambda l: l[0]['bbox'][1])
    #total_lines = np.append(total_lines, lines)

    na = np.array(image)
    # Iterate over traces to draw them on image.
    for l in traces:
        for i, t in enumerate(l):
            draw_img.line(t, fill='red', width=3)
            if i >= 1:
                draw_img.line([l[i-1][1], t[0]], fill='tomato', width=3)
    # Iterate over lines.
    for line in lines:
        endLine = False
        start = 0
        # In this loop, we aim to create multiple images for a row. Cuts are done randomly between characters.
        # While the line is not finished.
        while not endLine and len(line) >= MIN_CHAR:
            # Get random start and end indexes to cut a subimage in a row.
            end = np.random.randint(start + MIN_CHAR, start + MAX_CHAR)
            if end >= len(line):
                endLine = True
                end = len(line)
                if end - start < MIN_CHAR:
                    start = np.random.randint(max(0, end - MAX_CHAR), end - MIN_CHAR)
            
            # Get line and coord of subimage in row image.
            l = line[start:end]
            xmin = min(l, key=lambda c: c['bbox'][0])['bbox'][0]
            ymin = min(l, key=lambda c: c['bbox'][1])['bbox'][1]

            xmax = max(l, key=lambda c: c['bbox'][0] + c['bbox'][2])
            xmax = xmax['bbox'][0]+xmax['bbox'][2]
            ymax = max(l, key=lambda c: c['bbox'][1] + c['bbox'][3])
            ymax = ymax['bbox'][1]+ymax['bbox'][3]
            
            # Draw rectangles to debug on papyri image.
            color = tuple(np.random.choice(range(256), size=3))
            draw_img.rectangle((xmin, ymin, xmax, ymax), fill=None, outline=color, width=3)
            # Condition to split between train and test data. (Based on line index, not on sub image to prevent from test-train overlap)
            if idx_line % 5 == 0:
                save_line((xmin, ymin), (xmax, ymax), image_clean, idx_test, dist='test')
                save_incsv(idx_test, l, img_path, test_writer)
                idx_test+=1
            else:
                save_line((xmin, ymin), (xmax, ymax), image_clean, idx_train)
                save_incsv(idx_train, l, img_path, train_writer)
                idx_train+=1
            
            start = end-2
        idx_line += 1

    if SAVE_FULL_IMAGE:  
        image.save(os.path.join(output_lines, f'{idx}_{os.path.basename(img_path)}'), quality=100)
    image.close()
print(idx, (idx_train, idx_test))
ftrain.close()
ftest.close()
#dict_keys(['annotations', 'categories', 'database', 'images', 'licenses', 'texts'])

print("Done!")