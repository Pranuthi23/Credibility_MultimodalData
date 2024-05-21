""" Implements dataloaders for cub_200_2011 dataset"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
# import cv2
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
from torchvision import transforms
from PIL import Image
import numpy as np


class CUBDataset(Dataset):
    """Implements a torch Dataset class for the cub_200_2011 dataset."""

    def __init__(self, data_dir, train):
        """Initialize CUBDataset object.

        Args:
            data_dir (str): Directory of data.
            train (bool): Indicates whether to load training data or test data.
        """
        self.data_loc = data_dir
        self.train = train
        self.image_loc = os.path.join(data_dir, "images.txt")
        self.label_loc = os.path.join(data_dir, "image_class_labels.txt")
        self.train_test_split_loc = os.path.join(data_dir, "train_test_split.txt")
        self.att_file_loc = os.path.join(data_dir, "attributes.txt")
        self.image_attr_file_loc = os.path.join(data_dir, "attributes/image_attribute_labels.txt")

            
        
    
        #load image data
        images = pd.read_csv(self.image_loc, sep=' ', names=['img_id', 'file_path'])
        image_class_labels = pd.read_csv(self.label_loc, sep= ' ', names= ['img_id', 'target'])
        train_test_split = pd.read_csv(self.train_test_split_loc, sep= ' ', names= ['img_id', 'is_training'])

        image_data = images.merge(image_class_labels, on = 'img_id')
        image_data = image_data.merge(train_test_split, on = 'img_id')
        


        #load tabular data for image attributes
        attr_file = pd.read_csv(self.att_file_loc, sep=' ', names=["attr_no", "attr_value"])
        attr_file[['attr_name', 'attr_value']] = attr_file['attr_value'].str.split("::", expand = True)
        img_attr_file = pd.read_csv(self.image_attr_file_loc, sep=' ', names = ['img_id','attr_no','truth_value','d1','d2'])
        img_attr_file = img_attr_file.merge(attr_file, on= 'attr_no',validate="many_to_one")

        filtered_df = img_attr_file.loc[img_attr_file.groupby(['img_id','attr_name'])['truth_value'].idxmax()]
        filtered_df.loc[filtered_df['truth_value'] == 0, 'attr_value'] = 'None'
        tabular_data = filtered_df.pivot(index='img_id', columns='attr_name', values='attr_value')


        # converting to categorical codes
        tabular_data = tabular_data.apply(lambda col: pd.Categorical(col).codes)
        
        self.attr = tabular_data.columns.values.tolist()
        self.data = image_data.merge(tabular_data, on = 'img_id')

        #splitting data based on train_test_split.txt
        if self.train:
            self.data = self.data[self.data.is_training == 1]  #train data

            #Resizing to match GoogLeNet input
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.6983, 0.3918, 0.4474],
                    std=[0.1648, 0.1359, 0.1644],
                    ),
                ])
        else:
            self.data = self.data[self.data.is_training == 0]  #test data

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ])


    

    def __getitem__(self, index):
        """Get item from dataset.

        Args:
            index (int): Index of data to get

        Returns:
            tuple: Tuple of image input, tabular input and label
        """
        sample = self.data.iloc[index]
        image_data_path = os.path.join(self.data_loc,"images", sample.file_path)
        image = Image.open(image_data_path)
        image = self.transform(image).resize_(3,256, 384) 
        tabular = np.array(self.data[self.attr].iloc[index])
        label = sample.target - 1

        return image, tabular, label




    def __len__(self):
        """Get length of dataset."""
        return len(self.data)
    






def get_dataloader(data_dir, batch_size=40, num_workers=8, train_shuffle=True):
    """Get dataloaders for cub dataset.
    
    Args:
        data_dir (str): Directory of data.
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
    
    Returns:
        tuple: Tuple of (training dataloader, test dataloader)
        
    """

    dataset = CUBDataset(data_dir, True)

    train_data, val_data = random_split(dataset, [0.8, 0.2])

    val_dataloader = DataLoader(val_data, shuffle=False, num_workers= num_workers, batch_size=batch_size)
    
    train_dataloader = DataLoader(train_data, shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    
    test_dataloader = DataLoader(CUBDataset(data_dir, False), shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_loader,vl, test_loader = get_dataloader(data_dir='/mnt/datasets/cub/CUB_200_2011')

    for (idx, batch) in enumerate(vl):
        print("Batch: ", idx)
        x1 = batch[0]
        x2 = batch[1]
        y = batch[2]
        print(x2.shape)
