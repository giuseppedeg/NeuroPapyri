import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import random_split
from torchvision.transforms import Resize, RandomGrayscale, Normalize
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from models import EmbeddingNetwork
from utils.utils import resnet18, targets, targets_maj, char_min_to_maj
import config as C
from dataset import Dataset_ci



parser = argparse.ArgumentParser(
                    prog='Train_ci',
                    description='Train model for character identification')

parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-lr', '--lr', type=float, default=0.00005)
parser.add_argument('-d_lr_f', '--decraese_lr_factor', type=float, default=1)
parser.add_argument('-d_lr_e', '--decraese_lr_epoch', type=int, default=10)
parser.add_argument('-batch', '--batch_size', type=int, default=16)
parser.add_argument('-r', '--resume', type=str)
parser.add_argument('-s', '--step', type=int, default=1)
parser.add_argument('-d_path', '--data_path', type=str, default=C.ICDAR_DATASET02_PATH)
parser.add_argument('-s_train', '--split_train', type=float, default=30720)
parser.add_argument('-s_valid', '--split_valid', type=int, default=2048)
parser.add_argument('-r_h', '--resize_height', type=int, default=492)
parser.add_argument('-r_w', '--resize_width', type=int, default=1232)
parser.add_argument('-p', '--padding_transcription', type=int, default=None)
parser.add_argument('-t', '--targets', type=str, choices=["maj", "min"], default="min")
parser.add_argument('--output_save', type=str, default='/output/ci/')
parser.add_argument('--save_epochs', type = int, default = 1)

args = parser.parse_args()



def train(model, dataloader, criterion, optimizer):
    '''
    Train a model for one epoch
    
    Args:
        model (nn.Module): Model to train.
        dataloader (Dataloader): Dataloader used to get data to train.
        criterion (Loss): Object to compute the loss to train.
        optimizer (Optimizer): Optimizer used to train the model.
    '''
    model.train()
    total_loss = 0
    for image, gt, _, _ in tqdm(dataloader, desc="Batch", leave=False):
        image, gt = image.to(device), gt.to(device)
        image, gt = Variable(image), Variable(gt)
        
        optimizer.zero_grad()
        
        out, _, _ = model(image)
        out = out.transpose(0, 1)
        
        loss = criterion(out, gt)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss/len(dataloader)

def valid(model, dataloader, criterion, epoch):
    '''
    Valid function to compute the loss on the current model.
    Args:
        model (nn.Module): Model to train.
        dataloader (Dataloader): Dataloader used to get data to valid.
        criterion (Loss): Object to compute the loss on valid dataset.
    '''
    model.eval()
    good_pred = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for image, gt, transcription, _ in tqdm(dataloader, desc='Valid', leave=False):
            image, gt = image.to(device), gt.to(device)
            image, gt = Variable(image), Variable(gt)
            
            out, _, attn = model(image)
            out = out.transpose(0, 1)
            attn = attn.transpose(0, 1)
            
            loss = criterion(out, gt)
            
            out_flatten = out.flatten()
            gt_flatten = gt.flatten()
            
            fn = lambda x: 1 if x >= 0.5 else 0
            
            for i in range(len(out_flatten)):
                if fn(out_flatten[i]) == gt_flatten[i]:
                    good_pred += 1
                total += 1
            
            total_loss += loss.item()
    return good_pred / total, total_loss/len(dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

start_time = datetime.now().strftime("%Y%m%d_%H%M%S")


prefix = str(args.step) + ('R' if args.resume else '')
output_save = os.path.join(args.output_save, f'{prefix}_{start_time}')
os.makedirs(output_save)

if args.targets == "min":
     targets = targets
elif args.targets == "maj":
    targets = targets_maj

backbone = resnet18()
model = EmbeddingNetwork(backbone, targets, 512, args.batch_size, device)

if args.resume != None:
    model.load_state_dict(torch.load(args.resume))

model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# dataloader
if args.resize_height and args.resize_width:
    transforms = [Resize((args.resize_height, args.resize_width)), RandomGrayscale(p=0.2)]
    #transforms = [Resize((args.resize_height, args.resize_width)), RandomGrayscale(p=0.2), Normalize(mean=C.ICDAR_DATA_MEAN, std=C.ICDAR_DATA_STD)]

    dataset = Dataset_ci(args.data_path, transformer=transforms, padding=args.padding_transcription)
else:
    dataset = Dataset_ci(args.data_path)
tr_size = int(len(dataset)*args.split_train)
va_size = len(dataset) - tr_size
print(f"training images:{tr_size}, Validation images:{va_size}, Dataset images:{len(dataset)}")
train_set, valid_set = random_split(dataset, [tr_size, va_size])

train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

train_losses = []
valid_losses = []
f = open(os.path.join(output_save, f"log.txt"), 'w+', buffering=1)
f.write(f"epoch,train_loss,valid_loss,valid_acc\n")
for i in tqdm(range(args.epochs), desc="Epoch"):
    train_loss = train(model, train_dataloader, criterion, optimizer)
    acc_valid, valid_loss = valid(model, valid_dataloader, criterion, i)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    # Create a plot to vizualise the loss
    plt.plot([j+1 for j in range(i+1)], train_losses, color='r', label='Train')
    plt.plot([j+1 for j in range(i+1)], valid_losses, color='b', label='Valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_save, f"loss.jpg"), bbox_inches='tight')
    plt.close()
    
    print(f"Epoch {i+1}/{args.epochs} - Train loss: {train_loss} - Valid loss: {valid_loss} - Acc valid: {acc_valid}")
    f.write(f"{i+1},{train_loss},{valid_loss},{acc_valid}\n")
    if (i+1) % args.save_epochs == 0:
        torch.save(model.state_dict(), os.path.join(output_save, f"model_{i+1}.pth"))
f.close()
