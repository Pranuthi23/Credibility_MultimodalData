import os.path
import random

import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from torchvision.datasets.folder import make_dataset
import sys
from torch.utils.data import DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import numpy as np
from collections import Counter
from torchvision.transforms import functional as F
import copy


class AlignedConcDataset:

    def __init__(self,  FINE_SIZE,LOAD_SIZE, data_dir=None, transform=None, labeled=True):
        # self.cfg = cfg
        self.transform = transform
        self.data_dir = data_dir
        self.labeled = labeled
        self.LOAD_SIZE =LOAD_SIZE
        self.FINE_SIZE = FINE_SIZE

        self.classes, self.class_to_idx = find_classes(self.data_dir)
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.imgs = make_dataset(self.data_dir, self.class_to_idx, 'png')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        if w2 > self.FINE_SIZE:
            A = AB_conc.crop((0, 0, w2, h)).resize((self.LOAD_SIZE, self.LOAD_SIZE), Image.BICUBIC)
            B = AB_conc.crop((w2, 0, w, h)).resize((self.LOAD_SIZE, self.LOAD_SIZE), Image.BICUBIC)
        else:
            A = AB_conc.crop((0, 0, w2, h))
            B = AB_conc.crop((w2, 0, w, h))

        if self.labeled:
            sample = {'A': A, 'B': B, 'img_name': img_name, 'label': label}
        else:
            sample = {'A': A, 'B': B, 'img_name': img_name}

        if self.transform:
            sample['A'] = self.transform(sample['A'])
            sample['B'] = self.transform(sample['B'])

        # print( "Image: ", sample['img_name'], "label: ", sample['label'])
        return sample['A'], sample['B'], sample['label']
        # return sample


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        if self.padding > 0:
            A = F.pad(A, self.padding)
            B = F.pad(B, self.padding)

        # pad the width if needed
        if self.pad_if_needed and A.size[0] < self.size[1]:
            A = F.pad(A, (int((1 + self.size[1] - A.size[0]) / 2), 0))
            B = F.pad(B, (int((1 + self.size[1] - B.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and A.size[1] < self.size[0]:
            A = F.pad(A, (0, int((1 + self.size[0] - A.size[1]) / 2)))
            B = F.pad(B, (0, int((1 + self.size[0] - B.size[1]) / 2)))

        i, j, h, w = self.get_params(A, self.size)
        sample['A'] = F.crop(A, i, j, h, w)
        sample['B'] = F.crop(B, i, j, h, w)

        # _i, _j, _h, _w = self.get_params(A, self.size)
        # sample['A'] = F.crop(A, i, j, h, w)
        # sample['B'] = F.crop(B, _i, _j, _h, _w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.center_crop(A, self.size)
        sample['B'] = F.center_crop(B, self.size)
        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        if random.random() > 0.5:
            A = F.hflip(A)
            B = F.hflip(B)

        sample['A'] = A
        sample['B'] = B

        return sample


def find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class Resize(transforms.Resize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        h = self.size[0]
        w = self.size[1]

        sample['A'] = F.resize(A, (h, w))
        sample['B'] = F.resize(B, (h, w))

        return sample


class ToTensor(object):
    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        # if isinstance(sample, dict):
        #     for key, value in sample:
        #         _list = sample[key]
        #         sample[key] = [F.to_tensor(item) for item in _list]

        sample['A'] = F.to_tensor(A)
        sample['B'] = F.to_tensor(B)

        return sample


class Normalize(transforms.Normalize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.normalize(A, self.mean, self.std)
        sample['B'] = F.normalize(B, self.mean, self.std)

        return sample


class Lambda(transforms.Lambda):

    def __call__(self, sample):
        return self.lambd(sample)
    



def get_dataloader(data_dir,FINE_SIZE, LOAD_SIZE, batch_size=40, num_workers=8, train_shuffle=True):
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]
    train_transforms = list()
    train_transforms.append(transforms.Resize((LOAD_SIZE, LOAD_SIZE)))
    train_transforms.append(transforms.RandomCrop((FINE_SIZE, FINE_SIZE)))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(mean=mean, std=std))
    val_transforms = list()
    val_transforms.append(transforms.Resize((FINE_SIZE, FINE_SIZE)))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.Normalize(mean=mean, std=std))

    test_set = AlignedConcDataset(FINE_SIZE = FINE_SIZE, LOAD_SIZE = LOAD_SIZE, data_dir=os.path.join(data_dir, 'test'), 
                                                transform=transforms.Compose(val_transforms))

    train_loader = DataLoader(AlignedConcDataset(FINE_SIZE = FINE_SIZE, LOAD_SIZE = LOAD_SIZE, data_dir=os.path.join(data_dir, 'train'), 
                                                 transform=transforms.Compose(train_transforms)),batch_size=batch_size, 
                                                 shuffle=train_shuffle, num_workers=num_workers)
    
    test_loader = DataLoader(test_set, batch_size=len(test_set),
                                                shuffle=False, num_workers=num_workers)
    
    val_loader = DataLoader(AlignedConcDataset(FINE_SIZE = FINE_SIZE, LOAD_SIZE = LOAD_SIZE, data_dir=os.path.join(data_dir, 'val'), 
                                                transform=transforms.Compose(val_transforms)), batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)


    return train_loader, val_loader, test_loader


