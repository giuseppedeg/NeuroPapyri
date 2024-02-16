import os
from datetime import datetime
import time
import torch
from torch.utils.data import random_split
from torchvision.transforms import Resize, ColorJitter, Normalize
from utils.BinarizeTransform import BinarizeTransform
import numpy as np
from dataset import TwoLossDataset
from torch.autograd import Variable
from utils.utils import resnet18, targets
from models import EmbeddingNetwork, FreezeEmbeddingNetwork
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm
from utils.utils_triplet_loss import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from utils.utils_triplet_loss import BalancedLimitedBatchSampler, OnlineTripletLoss, cosine_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import config as C


parser = argparse.ArgumentParser(
                    prog='Train_dr',
                    description='Train model for document retrieval')

parser.add_argument('-d_path', '--data_path', type=str, default=C.ICDAR_DATASET02_PATH)
parser.add_argument('--output_save', type=str, default='output/dr/twolosses/')
parser.add_argument('-c', '--continue_training_model', type=str, help="model to use as a starting point for the training")
parser.add_argument('-r', '--pretrained_model', type=str, help="The pretrained model")
parser.add_argument('-tt', '--target_type', type=str, default="document", choices=["document", "author"])
parser.add_argument('-f', '--freeze', type=str, help="If True, the backbone of the model (up to the end of the attention block) is freeze")
parser.add_argument('-s_train', '--split_train', type=float, default=0.9, help="Percentage of training test. The remaining is used as validation set")
parser.add_argument('-r_h', '--resize_height', type=int, default=246)
parser.add_argument('-r_w', '--resize_width', type=int, default=616)
parser.add_argument('-ns', '--n_samples', type=int, default=2)
parser.add_argument('-bs', '--batch_size', type=int, default=64)
parser.add_argument('-lr', '--lr', type=float, default=0.0001)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-se', '--save_epochs', type=int, default=5)
parser.add_argument('-m', '--margin_loss', type=float, default=0.8)
parser.add_argument('-lw1', '--loss_weight1', type=float, default=0.25, help="is the weight for the attention loss, the weight w2 for the triplet loss will be calculated as w2=1-w1")

args = parser.parse_args()

data_path = args.data_path
output_save = args.output_save
continue_training_model = args.continue_training_model
pretrained_model=args.pretrained_model
target_type = args.target_type
freeze = False
if (args.freeze is not None) and (args.freeze.lower() != "false"):
    freeze = True
split_train = args.split_train
resize_height = args.resize_height
resize_width = args.resize_width
n_samples = args.n_samples #
max_batch_size = args.batch_size
lr = args.lr
end_epoch = args.epochs
save_epoch = args.save_epochs
margin_loss = args.margin_loss
loss_w1 = args.loss_weight1
loss_w2 = 1-loss_w1

num_workers = 1
pin_memory = True
#dist_fn = torch.nn.PairwiseDistance(p=2)
dist_fn = cosine_distance
backbone = resnet18()

seed = C.SEED_01
data_mean =  C.ICDAR_DATA_MEAN
data_std = C.ICDAR_DATA_STD

#targets = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω']
#targets = ['α','ε','ι','ν','ο','σ','τ','ρ','γ','η']
#targets = ['α','ε','ι','ν','ο']
#targets = ['α','ε','ο']
targets = ['ε']




def train(model, dataloader, criterion1, criterion2, optimizer, logfile):
    '''
    Train a model for one epoch
    
    Args:
        model (nn.Module): Model to train.
        dataloader (Dataloader): Dataloader used to get data to train.
        criterion (Loss): Object to compute the loss to train.
        optimizer (Optimizer): Optimizer used to train the model.
    '''
    total_loss1 = 0
    total_loss2 = 0
    total_loss = 0
    model.train()

    losses1 = []
    losses2 = []
    losses = []

    log_interval = int(len(dataloader)/C.TQDM_DISP_FR)

    #for batch_idx, data in enumerate(tqdm(dataloader, desc='Train', leave=False, miniters=len(dataloader)/utils.tqdm_disp_per), maxinterval=len(dataloader)/utils.tqdm_disp_per):
    for batch_idx, data in enumerate(dataloader):
        img_tensor, author_id, papyrus_id, img_path, target_chars, transcript = data
        
        if target_type == "document":
            target = papyrus_id if len(papyrus_id) > 0 else None
        else:
            target = author_id if len(author_id) > 0 else None
        
        img_tensor, target, target_chars = img_tensor.to(device), target.to(device), target_chars.to(device)
        img_tensor, target = Variable(img_tensor), Variable(target)

        optimizer.zero_grad()
        
        out, embs = model(img_tensor) # out-> 512 feature map

        loss1 = criterion1(out, target_chars)
        
        if type(embs) not in (tuple, list):
            embs = (embs,)
        loss_inputs = embs

        if target is not None:
            target = (target,)
        loss_inputs += target
            
        loss_outputs = criterion2(*loss_inputs)
        loss2 = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        loss = loss_w1*loss1+loss_w2*loss2
        
        losses1.append(loss1.item())
        losses2.append(loss2.item())
        losses.append(loss.item())
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        message = 'Train: [{:.0f}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss: {:.6f}'.format(
                batch_idx/2 * len(data[0]), len(dataloader.dataset),
                100. * batch_idx/2 / len(dataloader), np.mean(losses1), np.mean(losses2), np.mean(losses))

        logfile.write(message+"\n")

        if batch_idx/2 % log_interval == 0:
            print(message)
        
    return total_loss1/len(losses1), total_loss2/len(losses2), total_loss/len(losses) 

def valid(model, dataloader, criterion1, criterion2, logfile):
    '''
    Valid function to compute the loss on the current model.
    
    Args:
        model (nn.Module): Model to train.
        dataloader (Dataloader): Dataloader used to get data to valid.
        criterion (Loss): Object to compute the loss on valid dataset.
    '''
    total_loss1 = 0
    total_loss2 = 0
    total_loss = 0
    model.eval()
    losses1 = []
    losses2 = []
    losses = []

    log_interval = int(len(dataloader)/C.TQDM_DISP_FR)

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            img_tensor, author_id, papyrus_id, img_path, target_chars, transcript = data

            if target_type == "document":
                target = papyrus_id if len(papyrus_id) > 0 else None
            else:
                target = author_id if len(author_id) > 0 else None
            
            img_tensor, target, target_chars = img_tensor.to(device), target.to(device), target_chars.to(device)
            
            out, embs = model(img_tensor) 

            loss1 = criterion1(out, target_chars)
                    
            if type(embs) not in (tuple, list):
                embs = (embs,)
            loss_inputs = embs

            if target is not None:
                target = (target,)
            loss_inputs += target
                
            loss_outputs = criterion2(*loss_inputs)
            loss2 = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            loss = loss_w1*loss1 + loss_w2*loss2

            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses.append(loss.item())
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss += loss.item()

            message = 'Valid: [{:.0f}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss: {:.6f}'.format(
                    batch_idx/2 * len(data[0]), len(dataloader.dataset),
                    100. * batch_idx/2 / len(dataloader), np.mean(losses1), np.mean(losses2), np.mean(losses))

            logfile.write(message+"\n")

            if batch_idx/2 % log_interval == 0:
                print(message)
        
    return total_loss1/len(losses1), total_loss2/len(losses2), total_loss/len(losses)

def save_params():
    with open(os.path.join(output_save, "params.txt"), "w") as params_file:
        params_file.write(f"start_epoch={start_epoch},\n")
        params_file.write(f"end_epoch={end_epoch},\n")
        params_file.write(f"save_epoch={save_epoch},\n")
        params_file.write(f"lr={lr},\n")
        params_file.write(f"max_batch_size={max_batch_size},\n")
        params_file.write(f"n_samples={n_samples},\n")
        params_file.write(f"num_workers={num_workers},\n")
        params_file.write(f"freeze={freeze},\n")
        params_file.write(f"pretrained_model={pretrained_model},\n")
        params_file.write(f"continue_training_model={continue_training_model},\n")
        params_file.write(f"data_path={data_path},\n")
        params_file.write(f"split_train={split_train},\n")
        params_file.write(f"resize_height={resize_height},\n")
        params_file.write(f"resize_width={resize_width},\n")
        params_file.write(f"margin_loss={margin_loss},\n")
        params_file.write(f"loss_w1={loss_w1},\n")
        params_file.write(f"loss_w2={loss_w2},\n")
        params_file.write(f"dist_fn={dist_fn},\n")
        params_file.write(f"seed={seed},\n")
        params_file.write(f"output_save={output_save},\n")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# OUTPUT AND LOG FILES -------------------------------------------------------------------------------------------
output_save = os.path.join(output_save, f'{start_time}')
os.makedirs(output_save)

f = open(os.path.join(output_save, f"log.txt"), 'w+', buffering=1)
f.write(f"epoch,train_loss1,train_loss2,train_loss,valid_loss1,valid_loss2,valid_loss\n")

f_train = open(os.path.join(output_save, f"log_train.txt"), 'w+', buffering=1)

f_val = open(os.path.join(output_save, f"log_val.txt"), 'w+', buffering=1)

# DATA ---------------------------------------------------------------------------------------------------------
transforms = [
    Resize((resize_height, resize_width)), 
    #BinarizeTransform(p=0.3),
    #ColorJitter(hue=.05, saturation=.05),
    Normalize(mean=data_mean, std=data_std)
    ]

dataset = TwoLossDataset(basepath=data_path,transformer=transforms, label=target_type)

tr_size = int(len(dataset)*split_train)
va_size = len(dataset) - tr_size
print(f"Training images:{tr_size}, Validation images:{va_size}, Total Images:{len(dataset)}")
train_set, valid_set = random_split(dataset, [tr_size, va_size], generator=torch.Generator().manual_seed(seed))

n_classes_train = len(torch.unique(train_set.dataset.labels[train_set.indices]))
n_classes_val = len(torch.unique(valid_set.dataset.labels[valid_set.indices]))

# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedLimitedBatchSampler(train_set.dataset.labels[train_set.indices], n_classes=n_classes_train, n_samples=n_samples, max_batch_size=max_batch_size)
valid_batch_sampler = BalancedLimitedBatchSampler(valid_set.dataset.labels[valid_set.indices], n_classes=n_classes_val, n_samples=n_samples, max_batch_size=max_batch_size)

kwargs = {'num_workers': num_workers, 'pin_memory': True}
online_train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_batch_sampler, **kwargs)
online_val_loader = torch.utils.data.DataLoader(valid_set, batch_sampler=valid_batch_sampler, **kwargs)

# MODEL -------------------------------------------------------------------------------------------------------

model_pre = EmbeddingNetwork(backbone, targets, 512, train_batch_sampler.batch_size, device)

if pretrained_model != None:
    model_pre.load_state_dict(torch.load(pretrained_model))
    print(f"Model Loaded: {pretrained_model}")

model = FreezeEmbeddingNetwork(model_pre, out_channels=512, freeze=freeze, twoloss=True)

model.to(device)


# TRAINING ----------------------------------------------------------------------------------------------------
loss_fn_1 = torch.nn.BCELoss()
#torh.nn.CrossEntropyLoss()
loss_fn_2 = OnlineTripletLoss(margin_loss, RandomNegativeTripletSelector(margin_loss), dist_fn)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, total_iters=1)

train_losses1 = []
train_losses2 = []
train_losses = []
valid_losses1 = []
valid_losses2 = []
valid_losses = []

# CONTINUE TRAINING ---------------------------------------------------------------
start_epoch = 0

if continue_training_model is not None:
    checkpoint = torch.load(continue_training_model)
    start_epoch = checkpoint['epoch']+1
    train_losses1 = checkpoint['train_losses1']
    train_losses2 = checkpoint['train_losses2']
    train_losses = checkpoint['train_losses']
    valid_losses1 = checkpoint['valid_losses1']
    valid_losses2 = checkpoint['valid_losses2']
    valid_losses = checkpoint['valid_losses']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Continue training from epoch: {start_epoch}")

# DAVE PARAMS FILE -------------------------------------------------------------
save_params()

## Train
for epoch in tqdm(range(start_epoch, end_epoch), desc='Train'):
    start = time.time()

    train_loss1, train_loss2, train_loss = train(model, online_train_loader, loss_fn_1, loss_fn_2, optimizer, f_train)
    curr_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    valid_loss1, valid_loss2, valid_loss = valid(model, online_val_loader, loss_fn_1, loss_fn_2, f_val)

    train_losses.append(train_loss)
    train_losses1.append(train_loss1)
    train_losses2.append(train_loss2)
    valid_losses.append(valid_loss)
    valid_losses1.append(valid_loss1)
    valid_losses2.append(valid_loss2)

    
    # Create a plot to vizualise the loss
    #loss1
    plt.plot([j+1 for j in range(epoch+1)], train_losses1, color='r', label='Train')
    plt.plot([j+1 for j in range(epoch+1)], valid_losses1, color='b', label='Valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss1")
    m = max(max(train_losses1), max(valid_losses1))
    plt.ylim(0, m*1.05)
    plt.suptitle(f'lr: {lr} - Margin: {margin_loss} - Optimizer: {type(optimizer).__name__}')
    plt.legend()
    plt.savefig(os.path.join(output_save, f"loss1.jpg"), bbox_inches='tight')
    plt.close()

    #loss2
    plt.plot([j+1 for j in range(epoch+1)], train_losses2, color='r', label='Train')
    plt.plot([j+1 for j in range(epoch+1)], valid_losses2, color='b', label='Valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss2")
    m = max(max(train_losses2), max(valid_losses2))
    plt.ylim(0, m*1.05)
    plt.suptitle(f'lr: {lr} - Margin: {margin_loss} - Optimizer: {type(optimizer).__name__}')
    plt.legend()
    plt.savefig(os.path.join(output_save, f"loss2.jpg"), bbox_inches='tight')
    plt.close()

    #loss
    plt.plot([j+1 for j in range(epoch+1)], train_losses, color='r', label='Train')
    plt.plot([j+1 for j in range(epoch+1)], valid_losses, color='b', label='Valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    m = max(max(train_losses), max(valid_losses))
    plt.ylim(0, m*1.05)
    plt.suptitle(f'lr: {lr} - Margin: {margin_loss} - Optimizer: {type(optimizer).__name__}')
    plt.legend()
    plt.savefig(os.path.join(output_save, f"loss.jpg"), bbox_inches='tight')
    plt.close()


    
    print(f"Epoch {epoch+1}/{end_epoch} - Train loss: {train_loss} - Valid loss: {valid_loss} - Lr:{curr_lr} -  Time:{time.time()-start}")
    f.write(f"{epoch+1},{train_loss1},{train_loss2},{train_loss},{valid_loss1},{valid_loss2},{valid_loss}\n")

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(f"CUDA Memory -- Reserved:{r}  Allocated:{a}")
    
    # Save the model every {args.save_epochs} epoch.
    if (epoch+1) % save_epoch == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses1': train_losses1,
            'train_losses2': train_losses2,
            'train_losses': train_losses,
            'valid_losses1': valid_losses1,
            'valid_losses2': valid_losses2,
            'valid_losses': valid_losses,
        }, os.path.join(output_save, f"model_{epoch+1}.pth"))

f.close()
f_train.close()
f_val.close()
