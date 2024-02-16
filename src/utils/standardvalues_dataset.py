import torch
from torchvision import transforms
from dataset import PapyrowDataset
from torch.utils.data import DataLoader
import matplotlib
from tqdm import tqdm
matplotlib.use('TkAgg',force=True)
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


print("Switched to:",matplotlib.get_backend())

matplotlib.pyplot.ion()

resize_height=246
resize_width=616

data_path = 'data/ICDAR_lines_3rd_docretr/'
data_path = 'data/ICDAR_lines_3rd_docretr_expanded/'

transform_img = [
    transforms.Resize((resize_height, resize_width)), 
    #transforms.ToTensor(),
    # here do not use transforms.Normalize(mean, std)
]

dataset = PapyrowDataset(basepath=data_path,transformer=transform_img)


image_data_loader = DataLoader(
  dataset, 
  batch_size=16, 
  shuffle=False, 
  num_workers=0
)


def mean_std(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, author_id, papyrus_id, img_path in tqdm(loader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean,std


mean, std = mean_std(image_data_loader)
print(f"\nDataset:{data_path}\n\nmean_{mean}\n std:{std} \n")


