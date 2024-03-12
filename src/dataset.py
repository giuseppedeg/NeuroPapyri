import torch
from torch import nn
from torchvision import transforms
import os
from PIL import Image
from glob import glob
import csv
from ast import literal_eval
import pickle
from utils.utils import targets_dico, targets as trgs


class Dataset_ci(nn.Module):
    def __init__(self, path, dtype='train', transformer=None, padding=None):
        '''
        Ddataset class for the Character Identification.

        Args:
            path (str): Path to data.
            dtype (str): Type of data to load (train or test). It's used to find data, like: "{path}/{dtype}".
            transformer (list): List of transformers to apply on data, like data augmentation.
            padding (int): If not None, padding is used at the end of transcription to made every tensor with same size. `padding` is the size that transcription will have.
        '''
        self.dtype = dtype
        self.basepath = os.path.join(path, dtype)
        self.padding = padding
        self.files, self.targets, self.transcriptions, self.original_images_paths = self.getdatas()
        self.padding = padding
        if transformer != None:
            tr = []
            tr_norm = None
            for t in transformer:
                if type(t) == transforms.transforms.Normalize:
                    tr_norm = t
                else:
                    tr.append(t)
            if tr_norm is not None:
                self.transform = transforms.Compose([
                    *tr,
                    transforms.ToTensor(),
                    tr_norm
                ])
            else:
                self.transform = transforms.Compose([
                    *tr,
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.ToTensor()
    
    def __getitem__(self, idx):
        file = self.files[idx]
        target = self.targets[idx]
        transcription = self.transcriptions[idx]
        original_image_path = self.original_images_paths[idx]
        path = os.path.join(self.basepath, file)
        img = Image.open(path)
        
        img_tensor = self.transform(img)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        img.close()
        # Apply padding if asked to.
        if self.padding:
            transcription.extend(['0'] * (self.padding - len(transcription)))
        
        transcription = "".join(transcription)
        
        return img_tensor, target_tensor, transcription, original_image_path
    
    def __len__(self):
        return len(self.files)
    
    def getdatas(self):
        # Load data
        files, targets, transcriptions, original_paths = [], [], [], []
        with open(os.path.join(self.basepath, 'datas.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                file = row[0]
                target = literal_eval(row[1])
                transcription = literal_eval(row[2])
                original_filepath = row[4]
                transcription = ['Σ' if x=='Ϲ' else x for x in transcription]
                target[17] = 1 if ('Σ' in transcription) or ('σ' in transcription) else 0
                files.append(file)
                targets.append(target)
                transcriptions.append(transcription)
                original_paths.append(original_filepath)
        return files, targets, transcriptions, original_paths


class Dataset_dr(nn.Module):
    def __init__(self, basepath='.', transformer=None, dtype='train', label="document"):
        '''
        Papyrow dataset class.

        Args:
            basepath (str): Path to data.
            dtype (str): Type of data to load (train or test). It's used to find data, like: "{path}/{dtype}".
            transformer (list): List of transformers to apply on data, like data augmentation.
            label (str): 'document' or 'author'
        '''
        self.basepath = basepath
        self.dtype = dtype
        if transformer != None:
            tr = []
            tr_norm = None
            for t in transformer:
                if type(t) == transforms.transforms.Normalize:
                    tr_norm = t
                else:
                    tr.append(t)
            if tr_norm is not None:
                self.transform = transforms.Compose([
                    *tr,
                    transforms.ToTensor(),
                    tr_norm
                ])
            else:
                self.transform = transforms.Compose([
                    *tr,
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.ToTensor()
        self.label_type = label
        
        self.imgs, self.authors, self.labels = self.read_images()
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        item = self.imgs[idx]
        author = item['author']
        papyrus_id = item['papyrus_id']
        img_path = item['img']
        
        author_id = item['author_id']
        #author_id = self.authors.index(author)
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        img.close()
        
        # Some papyri has a format as "Victor_11-2", it's a fix to be sure that we get the papyri id.
        if '-' in papyrus_id:
            papyrus_id = papyrus_id.split('-')[0]
        
        return img_tensor, int(author_id), int(papyrus_id), img_path

    def read_images(self):
        '''
        Read CSV to get every images from dataset.
        '''
        images = []
        authors = []
        labels = []
        nb_img = 0
        with open(os.path.join(self.basepath, f'{self.dtype}.csv'), newline='') as csvfile:
            c = csv.reader(csvfile, delimiter=',')
            next(c)
            for line in c:
                for img_path in glob(os.path.join(self.basepath, line[3], '*')):
                    images.append({'author': line[0], 'papyrus_id': line[1], 'author_id': line[2], 'img': img_path})
                    if self.label_type == "document":
                        if "-" in line[1]:
                            labels.append(int(line[1].split('-')[0]))
                        else:
                            labels.append(int(line[1]))
                    else:
                        # author
                        labels.append(int(line[2]))
                        
                authors.append(line[0])     
        return images, list(set(authors)), torch.IntTensor(labels)


class TwoLossDataset(nn.Module):
    def __init__(self, get_triplet=False, basepath='../../papyri_1200/croppedImages', transformer=None, dtype='train', label="document"):
        '''
        Papyrow dataset class.

        Args:
            basepath (str): Path to data.
            dtype (str): Type of data to load (train or test). It's used to find data, like: "{path}/{dtype}".
            transformer (list): List of transformers to apply on data, like data augmentation.
            label (str): 'document' or 'author'
        '''
        self.get_triplet = get_triplet
        self.triplete_map = None
        self.basepath = basepath
        self.dtype = dtype
        if transformer != None:
            tr = []
            tr_norm = None
            for t in transformer:
                if type(t) == transforms.transforms.Normalize:
                    tr_norm = t
                else:
                    tr.append(t)
            if tr_norm is not None:
                self.transform = transforms.Compose([
                    *tr,
                    transforms.ToTensor(),
                    tr_norm
                ])
            else:
                self.transform = transforms.Compose([
                    *tr,
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.ToTensor()
        self.label_type = label
        
        self.imgs, self.authors, self.labels = self.read_images()
        self.imgs_target, self.imgs_transcript = self.read_targets_char()

        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        if self.get_triplet:
            assert self.triplete_map is not None, "Need first to compute the triplets map"

            item = self.imgs[idx]
            author = item['author']
            papyrus_id = item['papyrus_id']
            img_path = item['img']

            pos_path, neg_path = self.triplete_map[img_path]
            
            author_id = self.authors.index(author)
            img_anc = Image.open(img_path).convert('RGB')
            img_anc_tensor = self.transform(img_anc)
            img_anc.close()

            img_pos = Image.open(pos_path).convert('RGB')
            img_pos_tensor = self.transform(img_pos)
            img_pos.close()

            img_neg = Image.open(neg_path).convert('RGB')
            img_neg_tensor = self.transform(img_neg)
            img_neg.close()
            
            # Some papyri has a format as "Victor_11-2", it's a fix to be sure that we get the papyri id.
            if '-' in papyrus_id:
                papyrus_id = papyrus_id.split('-')[0]
            
            return img_anc_tensor, img_pos_tensor, img_neg_tensor, author_id, int(papyrus_id), (img_path, pos_path, neg_path)
        
        else:
            # not triplet
            item = self.imgs[idx]
            author = item['author']
            papyrus_id = item['papyrus_id']
            img_path = item['img']
            
            author_id = item['author_id']
            #author_id = self.authors.index(author)

            or_img = os.sep.join(os.path.normpath(img_path).split(os.sep)[-2:])

            target = self.imgs_target[or_img]
            # target = target.replace(" ", "")
            # target = target.replace("[", "")
            # target = target.replace("]", "")
            # target = target.split(',')
            # target = list(map(int, target))
            transcript = self.imgs_transcript[or_img]

            target_tensor = torch.tensor(target, dtype=torch.float32)

            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            img.close()
            
            # Some papyri has a format as "Victor_11-2", it's a fix to be sure that we get the papyri id.
            if '-' in papyrus_id:
                papyrus_id = papyrus_id.split('-')[0]
            
            return img_tensor, int(author_id), int(papyrus_id), img_path, target_tensor, transcript
    
    def read_images(self):
        '''
        Read CSV to get every images from dataset.
        '''
        images = []
        authors = []
        labels = []
        nb_img = 0
        with open(os.path.join(self.basepath, f'{self.dtype}.csv'), newline='') as csvfile:
            c = csv.reader(csvfile, delimiter=',')
            next(c)
            for line in c:
                for img_path in glob(os.path.join(self.basepath, line[3], '*')):
                    images.append({'author': line[0], 'papyrus_id': line[1], 'author_id': line[2], 'img': img_path})
                    if self.label_type == "document":
                        if "-" in line[1]:
                            labels.append(int(line[1].split('-')[0]))
                        else:
                            labels.append(int(line[1]))
                    else:
                        # author
                        labels.append(int(line[2]))
                        
                authors.append(line[0])     
        return images, list(set(authors)), torch.IntTensor(labels)
     
    def set_triplet_map(self, triplete_map):
        self.triplete_map = triplete_map

    def read_targets_char(self):
        '''
        Read pickle files to get every target images
        '''
        targets = {}
        transcript = {}
        with open(os.path.join(self.basepath, f'{self.dtype}_target'), 'rb') as file:
            data = pickle.load(file)

            for img, dir_tar in data.items():
                targets[img] = dir_tar['target']
                transcript[img] = dir_tar['transcript']

        return  targets, transcript
    
    
    def set_triplet_map(self, triplete_map):
        self.triplete_map = triplete_map
                
