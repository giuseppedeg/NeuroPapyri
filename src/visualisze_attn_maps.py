import os
import torch
from torchvision import transforms
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops 
from models import EmbeddingNetwork, FreezeEmbeddingNetwork
from utils.utils import resnet18, targets, targets_maj, char_min_to_maj
from dataset import Dataset_dr
import matplotlib
matplotlib.use('Agg')
import argparse
import config as C
from tqdm import tqdm


parser = argparse.ArgumentParser(
                    prog='Test_dr',
                    description='Test model for document retrieval')

parser.add_argument('-d_path', '--data_path', type=str, default=C.ICDAR_DATASET02_PATH)
parser.add_argument('-o', '--output_save', type=str, default='results/dr/attentionmaps')
parser.add_argument('-m','--model', help='Model to test', required=True)
parser.add_argument('-r_h', '--resize_height', type=int, default=246)
parser.add_argument('-r_w', '--resize_width', type=int, default=616)
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-i', '--images_to_test', type=int, default=25)

args = parser.parse_args()

basepath_dataset = args.data_path
test_model_path = args.model
out_folder = args.output_save
r_height = args.resize_height
r_width = args.resize_width
img_to_save = args.images_to_test


#targets = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω']
#targets = ['α','ε','ι','ν','ο','σ','τ','ρ','γ','η']
#targets = ['α','ε','ι','ν','ο']
#targets = ['α','ε','ο']
targets = ['ε']

random.seed(C.SEED_01)

on_screen_images = False
if not on_screen_images:
    plt.ioff()

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(test_model_path)
backbone = resnet18()
model_pre = EmbeddingNetwork(backbone, targets, 512, 1, device)
model_freeze = FreezeEmbeddingNetwork(model_pre, 512)
model_freeze.load_state_dict(checkpoint['model_state_dict'])
model =  model_freeze.embedding_network
model.to(device)
model.eval()

resizer = transforms.Resize((r_height, r_width))
dataset = Dataset_dr(dtype='test', basepath=basepath_dataset, transformer=[resizer])
to_tensor = transforms.ToTensor()


tr_gt = []
for _ in targets:
    tr_gt.append(1)

    
for i in tqdm(range(img_to_save)):
    n = random.randrange(len(dataset))

    im, _, _, _ = dataset[n]
    g = torch.tensor(tr_gt)
    #t = tr_transcriptions

    im = im.to(device)
    
    o, _, a = model(im.unsqueeze(0))
    o = o.transpose(0, 1)
    a = a.transpose(0, 1)

    o = o[0].cpu().detach().numpy()
    a = a[0].cpu().detach().numpy()
    g = g.cpu().detach().numpy()
    im = im.permute(1, 2, 0).cpu().numpy()
    
    #print(o, g)
    
    indexes = np.where(g == 1)[0]
    shape = im.shape
    size = (shape[1], shape[0])
    zeros = np.zeros([size[1], size[0], 2]).astype(np.uint8)
    cnt = (g == 1).sum()
    fig = plt.figure(figsize=(cnt+1,cnt*1.7), dpi=200)

    axs = fig.subplots(cnt+1, 1)
    axs[0].imshow(im)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    im = (im*255).astype(np.uint8)
    pil_img = Image.fromarray(im)
    for idx in range(cnt):
        curr_a = a[indexes[idx]]
        curr_a = cv2.resize(curr_a, size, interpolation = cv2.INTER_AREA)
        curr_a = (curr_a*255*2).astype(np.uint8)
        curr_a = curr_a[..., np.newaxis]
        curr_a = np.concatenate((curr_a, zeros), axis=2)

        a_img = Image.fromarray(curr_a, mode='RGB')

        composite = Image.blend(pil_img, a_img, 0.50)
        #composite = Image.blend(pil_img, a_img, 1)
       
        composite = ImageChops.add(pil_img, a_img) 
        
        axs[idx+1].imshow(composite)
        axs[idx+1].set_xticks([])
        axs[idx+1].set_yticks([])

        a_img.close()
        composite.close()

    pil_img.close()
    if on_screen_images:
        plt.show()
    fig.savefig(os.path.join(out_folder, f"visualize_{i}.jpg"), bbox_inches='tight')
    plt.close()
    
print("Done")