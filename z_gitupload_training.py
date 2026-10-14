from pipeline import XRS_Dataset_Sliding_Window_Timeseries
from models import  Encoder4Attn, AttnDecoder_ver3

import os, sys, gc
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import shutil

from tqdm import tqdm, trange
from collections import Counter
from torch.utils.data import DataLoader, ConcatDataset,Subset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 1881

model_name = "model_name"
data_path = "/datapath/"

##################

def _init_fn(seed):
    np.random.seed(seed)

encoded_size = 8
hidden_size = 1024
drop_ratio = 0.1
bidirectional = False
initialize = True
lr = 0.001

EPOCHS = 100
batch_size = 1200

Normalization = 'No_Normalization'
Resampling_mode = "Original"
data_type = ['xrsb']
num_features = len(data_type)

flare_classes = ['C','M', 'X']
hit_info = ["C", "MX", -5]
input_len = 60
folds_num = [1, 2, 3, 4]

for i in range(4):

    gc.collect()
    num_workers = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    train_folds = [folds_num[(i + j) % len(folds_num)] for j in range(1, len(folds_num))]
    test_fold = folds_num[i]

    ######## Training data
    Train_dataset = XRS_Dataset_Sliding_Window_Timeseries(root=data_path,
                                                          folds=train_folds,
                                                          flare_classes=flare_classes,
                                                          data_type=data_type,
                                                          logarithm=True)

    ######## Test data ########
    Test_dataset = XRS_Dataset_Sliding_Window_Timeseries(root=data_path, folds=[test_fold],
                                                         flare_classes=flare_classes,
                                                         data_type=data_type,
                                                         logarithm=True)
    ###################
    Train_data_loader = DataLoader(dataset=Train_dataset, batch_size=batch_size,
                                   num_workers=num_workers, shuffle=True,
                                   worker_init_fn=_init_fn(seed))


    Test_data_loader = DataLoader(dataset=Test_dataset, batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=False, worker_init_fn=_init_fn(seed))

    E = Encoder4Attn(input_size=num_features, hidden_size=hidden_size, layer_num=2,
                     encoded_size=encoded_size, bidirection=bidirectional,
                     drop_frac=drop_ratio, initiallize=initialize).to(device=DEVICE,
                                                                      dtype=torch.float32)
    D = AttnDecoder_ver3(input_len=input_len, hidden_size=hidden_size, layer_num=2,
                         encoded_size=encoded_size, drop_frac=drop_ratio,
                         initiallize=initialize).to(device=DEVICE, dtype=torch.float32)

    E_optim = torch.optim.Adam(E.parameters(), lr=lr, betas=[0.9, 0.99])
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=[0.9, 0.99])
    loss_fn = nn.MSELoss().to(device=DEVICE)

    # for epoch in trange(EPOCHS):
    for epoch in trange(EPOCHS):
        E.train()
        D.train()

        for Train_input, Train_target in Train_data_loader:
            Train_input = Train_input.to(device=DEVICE, dtype=torch.float32)
            Train_target = Train_target.to(device=DEVICE, dtype=torch.float32)

            Encoder_result, Encoder_hidden = E(Train_input)
            Input_lastpart = Train_input[:, -1:, :]
            Result, Decoder_hidden, _ = D(Input_lastpart, Encoder_hidden, Encoder_result)

            l = loss_fn(torch.squeeze(Result), torch.squeeze(Train_target))
            E_optim.zero_grad()
            D_optim.zero_grad()
            l.backward()
            E_optim.step()
            D_optim.step()

        with torch.no_grad():
            E.eval()
            D.eval()

            for Test_input, Test_target in Test_data_loader:
                Test_input = Test_input.to(device=DEVICE, dtype=torch.float32)
                Test_target = Test_target.to(device=DEVICE, dtype=torch.float32)

                Encoder_result, Encoder_hidden = E(Test_input)
                Input_lastpart = Test_input[:, -1:, :]
                Result, Decoder_hidden, _ = D(Input_lastpart, Encoder_hidden,
                                              Encoder_result)