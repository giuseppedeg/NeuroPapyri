import os
import torch
from torchvision.transforms import ToPILImage
from torch.nn import Conv2d, functional as F, Sequential
from torchvision.models import resnet18 as _resnet18

import numpy as np
from numpy import dot
from numpy.linalg import norm

# Targets used to learn attention maps and embeddings.
targets = np.array([
    'α',
    'β',
    'γ',
    'δ',
    'ε',
    'ζ',
    'η',
    'θ',
    'ι',
    'κ',
    'λ',
    'μ',
    'ν',
    'ξ',
    'ο',
    'π',
    'ρ',
    'σ',
    'τ',
    'υ',
    'φ',
    'χ',
    'ψ',
    'ω'
])

targets_maj = np.array([
    'Α',
    'Β',
    'Γ',
    'Δ',
    'Ε',
    'Ζ',
    'Η',
    'Θ',
    'Ι',
    'Κ',
    'Λ',
    'Μ',
    'Ν',
    'Ξ',
    'Ο',
    'Π',
    'Ρ',
    'Σ',
    'Τ',
    'Υ',
    'Φ',
    'Χ',
    'Ψ',
    'Ω'
])

targets_dico = {
    'Alpha': 'α',
    'Beta': 'β',
    'Gamma': 'γ',
    'Delta': 'δ',
    'Epsilon': 'ε',
    'Zeta': 'ζ',
    'Eta': 'η',
    'Theta': 'θ',
    'Iota': 'ι',
    'Kappa': 'κ',
    'Lambda': 'λ',
    'Mu': 'μ',
    'Nu': 'ν',
    'Xi': 'ξ',
    'Omicron': 'ο',
    'Pi': 'π',
    'Rho': 'ρ',
    'Sigma': 'σ',
    'Tau': 'τ',
    'Upsilon': 'υ',
    'Phi': 'φ',
    'Khi': 'χ',
    'Psi': 'ψ',
    'Omega': 'ω',
}

def char_min_to_maj(char):
    return targets_maj[np.where(targets == char)][0]

def char_maj_to_min(char):
    return targets[np.where(targets_maj == char)][0]

# Compute true positive.
def tp(out, gt):
    n = 0
    for i in range(len(out)):
        if out[i] == gt[i] and out[i] == 1:
            n +=1
    return n

# Compute true negative.
def tn(out, gt):
    n = 0
    for i in range(len(out)):
        if out[i] == gt[i] and out[i] == 0:
            n +=1
    return n

# Compute false negative.
def fn(out, gt):
    n = 0
    for i in range(len(out)):
        if out[i] != gt[i] and out[i] == 0:
            n +=1
    return n

# Compute false positive.
def fp(out, gt):
    n = 0
    for i in range(len(out)):
        if out[i] != gt[i] and out[i] == 1:
            n +=1
    return n


def save_model(model, path, epoch):
    '''
    Save a model.
    
    Args:
        model (nn.Module): Model to save.
        path (str): path used to save the model.
        epoch (int): epoch of the model usef in the file name.
    '''
    name = f"net_epoch_${epoch}.pth"
    path = os.path.join(path, name)
    torch.save(model.state_dict(), path)

def resnet18():
    '''
    Give a resnet18 model not train without last layers.
    '''
    model = _resnet18()
    model = Sequential(*(list(model.children())[:-2]))
    return model

def cos_sim(a, b):
    '''
    Compute cos similarity between array.
    
    Args:
        a (ndarray): Model to save.
        b (ndarray): path used to save the model.
    '''
    return dot(a, b)/(norm(a)*norm(b))

def hard_n(label, labels, n) -> bool:
    '''
    Return if hard-n is respect for the label and the labels list.
    
    Args:
        label (): label.
        labels (list): ordonate list of labels.
        n (int): to define the N of hard-n.
    '''
    for i in range(n):
        if label != labels[i]:
            return False
    return True

def soft_n(label, labels, n) -> bool:
    '''
    Return if soft-n is respect for the label and the labels list.
    
    Args:
        label (): label.
        labels (list): ordonate list of labels.
        n (int): to define the N of soft-n.
    '''
    for i in range(n):
        if label == labels[i]:
            return True
    return False
