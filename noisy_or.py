import sys
import os
sys.path.append(os.getcwd())
from MultiBench.unimodals.common_models import LeNet, MLP, Constant
from MultiBench.datasets.avmnist.get_data import get_dataloader
import torch
from torch import nn
from MultiBench.training_structures.unimodal import train, test
import numpy as np
from sklearn.metrics import accuracy_score




traindata, validdata, testdata = get_dataloader("Datasets/avmnist")



def image_model():
    modalnum = 0
    channels = 3
    
    encoder = LeNet(1, channels, 3).cuda()
    head = MLP(channels*8, 100, 10).cuda()


    train(encoder, head, traindata, validdata, 25, optimtype=torch.optim.Adam, lr=0.01, weight_decay=0.0001, modalnum=modalnum)

    print("Testing:")
    encoder = torch.load('encoder.pt').cuda()
    head = torch.load('head.pt')
    
    dic = test(encoder, head, testdata, modalnum=modalnum, no_robust=True)
    return dic









def audio_model():
    modalnum = 1
    channels = 6
    
    encoder = LeNet(1, channels, 5).cuda()
    head = MLP(channels*32, 100, 10).cuda()


    train(encoder, head, traindata, validdata, 25, optimtype=torch.optim.Adam, lr=0.01, weight_decay=0.0001, modalnum=modalnum)

    print("Testing:")
    encoder = torch.load('encoder.pt').cuda()
    head = torch.load('head.pt')
    
    dic = test(encoder, head, testdata, modalnum=modalnum, no_robust=True)
    return dic










def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()



def noisy_or(ex_image, ex_audio):
    prob = []
    for cls in range(ex_image.shape[0]):
        prob.append(1 - (1 - ex_image[cls]) * (1 - ex_audio[cls]))
    return np.argmax(prob)





def noisy_or_label(image_model, audio_model):
    noisy_or_prob = []
    for i in range(image_model.shape[0]):
        im = softmax(image_model[i])
        am = softmax(audio_model[i])
        noisy_or_prob.append(noisy_or(im, am))
    return noisy_or_prob





def imp_noisyor(Unimodal_image_model, Unimodal_audio_model):
#     Unimodal_audio_model = audio_model()
#     Unimodal_image_model = image_model()
    image_model = Unimodal_image_model['Pred Prob']
    audio_model = Unimodal_audio_model['Pred Prob']
    noisyor_label = noisy_or_label(image_model, audio_model)
    true_label = Unimodal_image_model['True Label']
    
    score = accuracy_score(noisyor_label, true_label)
    print("Accuracy of noisy-or :", score)
    
    
    

