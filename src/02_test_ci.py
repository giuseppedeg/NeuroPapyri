import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Resize, Normalize
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from models import EmbeddingNetwork
from utils.utils import resnet18, targets, targets_maj, char_min_to_maj, tp, tn, fp, fn
from dataset import Dataset_ci
import config as C


parser = argparse.ArgumentParser(
                    prog='Test_ci',
                    description='Test the model for character identification')

parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-d_path', '--data_path', type=str, default='data/ICDAR_lines_01')
parser.add_argument('-p', '--padding_transcription', type=int, default=10)
parser.add_argument('-r_h', '--resize_height', type=int, default=492)
parser.add_argument('-r_w', '--resize_width', type=int, default=1232)
parser.add_argument('--output_save', type=str, default='results/ci')
parser.add_argument('-batch', '--batch_size', type=int, default=10)

args = parser.parse_args()

def test(model, dataloader):
    gt_chars = np.zeros(24)
    fnd_chars = np.zeros(24)
    pred_true = 0
    tp_ = 0
    tn_ = 0
    fp_ = 0
    fn_ = 0
    for image, gt, _ ,_ in tqdm(dataloader, desc='Test', leave=False):
        image, gt = image.to(device), gt.to(device)
        image, gt = Variable(image), Variable(gt)
        
        out, _, _ = model(image)
        out = out.transpose(0, 1)
        
        fnc = lambda x: 1 if x >= 0.5 else 0
        
        for out_, gt_ in zip(out.cpu().detach().numpy(), gt.cpu().detach().numpy()):
            gt_chars += gt_
            out_fn = np.array([fnc(o) for o in out_])
            pred_true += out_fn.sum()
            fnd_chars = np.add(fnd_chars, np.logical_and(out_fn, gt_).astype(int))
            tp_ += tp(out_fn, gt_)
            tn_ += tn(out_fn, gt_)
            fp_ += fp(out_fn, gt_)
            fn_ += fn(out_fn, gt_)
    return gt_chars, fnd_chars, tp_, tn_, fp_, fn_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = resnet18()
model = EmbeddingNetwork(backbone, targets, 512, 4, device)
model_dict = torch.load(args.model)
model.load_state_dict(model_dict)

model.to(device)
model.eval()


transforms = [Resize((args.resize_height, args.resize_width))]
#transforms = [Resize((args.resize_height, args.resize_width)), Normalize(mean=C.ICDAR_DATA_MEAN, std=C.ICDAR_DATA_STD)]
dataset = Dataset_ci(args.data_path, dtype='test',transformer=transforms, padding=args.padding_transcription)
print(f"len test:{len(dataset)}")

test_dataloader = DataLoader(dataset, batch_size=4)

gt, fnd, tp, tn, fp, fn = test(model, test_dataloader)

fig, ax = plt.subplots()
ax.bar(targets, fnd, label='Identified')
ax.bar(targets, np.subtract(gt, fnd), bottom=fnd, label='Not Identified')
gt_sum = np.sum(gt)
fnd_sum = np.sum(fnd)

y_offset = -25
for bar in ax.patches:
   ax.text(
       bar.get_x() + bar.get_width() / 2,
       bar.get_height() + bar.get_y() + (-100 if bar.get_height() >= 100 else -80),
       round(bar.get_height()),
       ha='center',
       color='w',
       weight='bold',
       size=8,
       rotation='vertical'
   )

if not os.path.exists(args.output_save):
    os.mkdir(args.output_save)
recal = tp/(tp+fn)*100 
precision = tp/(tp+fp)*100
f1 = 2 * precision*recal / (precision + recal)
print(f"Precision:{round(precision,2)}  Recal:{round(recal,2)}  F1:{round(f1,2)}")
with open(os.path.join(args.output_save, "results.txt"), "w") as out_file:
    out_file.write(f"Results form model {args.model}\n\n")
    out_file.write(f"Precision:{round(precision,2)}\nRecal:{round(recal,2)}\nF1:{round(f1,2)}\n")
ax.set_title(f"Recal: {round(recal , 2)}% Precision: {round(precision, 2)}%")
ax.legend(loc='lower right')
fig.savefig(f"{os.path.join(args.output_save, 'bar_chart_ci.jpg')}", bbox_inches='tight')