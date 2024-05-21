import scipy.io as sio
import numpy as np
from numpy.random import shuffle
import math
import os

from torch.utils.data import Dataset, DataLoader, random_split



class CubDataSet(Dataset):

    def __init__(self, data, view_number, labels):
        """
        Construct a DataSet.
        """
        self.data = dict()
        self.num_examples = data[0].shape[0]
        self.labels = labels
        for v_num in range(view_number):
            self.data[str(v_num)] = data[v_num]

        self.labels = np.squeeze(self.labels)


    
    def __getitem__(self, index):
        image = self.data['0'][index]
        attr = self.data['1'][index]
        label = self.labels[index] - 1
        return image, attr, label

    
    def __len__(self):
        return self.num_examples



def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)




def read_data(str_name, ratio, Normal=1):
    """read data and spilt it train set and test set evenly
    :param str_name:path and dataname
    :param ratio:training set ratio
    :param Normal:do you want normalize
    :return:dataset and view number
    """
    data = sio.loadmat(str_name)
    view_number = data['X'].shape[1]
    X = np.split(data['X'], view_number, axis=1)
    X_train = []
    X_test = []
    labels_train = []
    labels_test = []
    if min(data['gt']) == 0:
        labels = data['gt'] + 1
    else:
        labels = data['gt']
    classes = max(labels)[0]
    all_length = 0
    for c_num in range(1, classes + 1):
        c_length = np.sum(labels == c_num)
        index = np.arange(c_length)
        shuffle(index)
        labels_train.extend(labels[all_length + index][0:math.floor(c_length * ratio)])
        labels_test.extend(labels[all_length + index][math.floor(c_length * ratio):])
        X_train_temp = []
        X_test_temp = []
        for v_num in range(view_number):
            X_train_temp.append(X[v_num][0][0].transpose()[all_length + index][0:math.floor(c_length * ratio)])
            X_test_temp.append(X[v_num][0][0].transpose()[all_length + index][math.floor(c_length * ratio):])
        if c_num == 1:
            X_train = X_train_temp
            X_test = X_test_temp
        else:
            for v_num in range(view_number):
                X_train[v_num] = np.r_[X_train[v_num], X_train_temp[v_num]]
                X_test[v_num] = np.r_[X_test[v_num], X_test_temp[v_num]]
        all_length = all_length + c_length
    if (Normal == 1):
        for v_num in range(view_number):
            X_train[v_num] = Normalize(X_train[v_num])
            X_test[v_num] = Normalize(X_test[v_num])


    traindata = CubDataSet(X_train, view_number, np.array(labels_train))
    testdata = CubDataSet(X_test, view_number, np.array(labels_test))
    return traindata, testdata


def get_dataloader(data_dir, train_shuffle = True, batch_size = 40, num_workers = 8):

    X_train, X_test = read_data(os.path.join(data_dir, 'cub_googlenet_doc2vec_c10.mat'), 0.8)
    X_train, X_val = random_split(X_train, [0.7,0.3])
    test_size = len(X_test)

    train_dataloader = DataLoader(X_train, batch_size= batch_size, shuffle= train_shuffle, num_workers= num_workers)
    val_dataloader = DataLoader(X_val, batch_size= batch_size, shuffle= False, num_workers= num_workers)
    test_dataloader = DataLoader(X_test, batch_size= test_size, shuffle= False, num_workers= num_workers)

    return train_dataloader, val_dataloader, test_dataloader





if __name__=='__main__':
    tl, vl, tel = get_dataloader('/mnt/datasets/cub_mini')
    trainiter = iter(tl)
    Images, Depths, labels = next(trainiter)
    print(Images.shape)
    print(((Depths.shape)))
    print((labels.shape))