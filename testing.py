from pipeline import XRS_Dataset_Sliding_Window_Timeseries
from models import Encoder4Attn, AttnDecoder_ver3, AttnDecoder_ver3_uncertainty

import os, sys
import gc
import glob
import torch
import numpy as np
import pandas as pd

from tqdm import trange
from torch.utils.data import DataLoader


##################
data_path = "data_path"
save_path = "save_path"

##################

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_workers = 0

###################

data_type = ['xrsb']
batch_size = 1
num_features = len(data_type)

flare_classes = ['C','M', 'X']

encoded_size = 8
hidden_size = 1024
mc_drop_ratio = 0.1
bidirectional = False
initialize = True
input_len = 60
T = 100

folds_num = [1, 2, 3, 4]

#########################


CV_Test_targets = []
CV_Test_results = []
CV_Test_classes = []
CV_Test_event_id = []
CV_Test_af_onset = []
CV_Test_bf_peak = []
CV_Test_fold = []
CV_Test_start_time = []
CV_Test_model_variance = []

for i in trange(4):
        
    gc.collect()

    train_folds = [folds_num[(i + j) % len(folds_num)] for j in range(1, len(folds_num))]
    test_fold = [folds_num[i]]

    model_encoder_path = f"\model_encoder_path\CV_{test_fold[0]}"
    model_decoder_path = f"\model_decoder_path\CV_{test_fold[0]}"

    model_encoder = glob.glob(model_encoder_path)[0]
    model_decoder = glob.glob(model_decoder_path)[0]    
        
    ######## Test data ########
    Test_dataset = XRS_Dataset_Sliding_Window_Timeseries(root = data_path, folds=test_fold, 
                                                           flare_classes=flare_classes, data_type=data_type, logarithm=True)
    ###################
    
    Test_data_loader = DataLoader(dataset = Test_dataset, batch_size = batch_size, num_workers = num_workers, 
                                  shuffle = False)

    E = Encoder4Attn(input_size=num_features, hidden_size=hidden_size, layer_num=2,
                     encoded_size=encoded_size, bidirection=bidirectional,
                     drop_frac=drop_ratio, initiallize=initialize).to(device=DEVICE, dtype=torch.float32)
    D = AttnDecoder_ver3(input_len=input_len, hidden_size=hidden_size, layer_num=2,
                         encoded_size=encoded_size, drop_frac=drop_ratio,
                         initiallize=initialize).to(device=DEVICE, dtype=torch.float32)
    D_uncertainty = AttnDecoder_ver3_uncertainty(input_len=input_len, hidden_size=hidden_size, layer_num=2,
                                                 encoded_size=encoded_size, drop_frac=drop_ratio, mc_dropout=mc_drop_ratio,
                                                 initiallize=initialize).to(device=DEVICE, dtype=torch.float32)

    E.load_state_dict(torch.load(model_encoder, weights_only=True))
    D.load_state_dict(torch.load(model_decoder, weights_only=True))
    D_uncertainty.load_state_dict(torch.load(model_decoder, weights_only=True))

    Test_targets = []
    Test_results = []
    Test_classes = []    

    Test_event_id = []
    Test_af_onset = []
    Test_bf_peak = []

    Test_start_time = []
    Test_model_variance = []

    with torch.no_grad():                                                
        E.eval()
        D.eval()
        D_uncertainty.eval()
        for Test_input, Test_target, Test_class, event_id, af_onset, bf_peak, start_time in Test_data_loader:
            
            Test_input  = Test_input.to(device=DEVICE, dtype=torch.float32)
            Test_target = Test_target.to(device=DEVICE, dtype=torch.float32)
    
            Encoder_result, Encoder_hidden = E(Test_input)
            Input_lastpart = Test_input[:, -1:, :]

            Result, Decoder_hidden, _ = D(Input_lastpart, Encoder_hidden, Encoder_result)
            
            Uncert_pred_results = []
            for _t in range(T):
                Result_uncert, Decoder_hidden, _ = D_uncertainty(Input_lastpart, Encoder_hidden, Encoder_result)
                Uncert_pred_results.append(Result_uncert.cpu().numpy().reshape(-1))
            Uncert_pred_results = np.concatenate(Uncert_pred_results, axis=0)
            Var_model = np.var(Uncert_pred_results)                

            Test_targets.append(Test_target.cpu().numpy().reshape(-1))
            Test_results.append(Result.cpu().numpy().reshape(-1))
            Test_classes.append(Test_class.cpu().numpy().reshape(-1))

            Test_event_id.append(event_id.cpu().numpy().reshape(-1))
            Test_af_onset.append(af_onset.cpu().numpy().reshape(-1))
            Test_bf_peak.append(bf_peak.cpu().numpy().reshape(-1))
            
            Test_start_time.append(start_time)
            Test_model_variance.append(Var_model)

    Test_targets = np.concatenate(Test_targets, axis=0)
    Test_results = np.concatenate(Test_results, axis=0)
    Test_classes = np.concatenate(Test_classes, axis=0)

    Test_event_id = np.concatenate(Test_event_id, axis=0)
    Test_af_onset = np.concatenate(Test_af_onset, axis=0)
    Test_bf_peak = np.concatenate(Test_bf_peak, axis=0)

    Test_start_time = np.concatenate(Test_start_time, axis=0)
    Test_model_variance = np.array(Test_model_variance)

    CV_Test_targets.append(Test_targets)
    CV_Test_results.append(Test_results)
    CV_Test_classes.append(Test_classes)
    CV_Test_event_id.append(Test_event_id)
    CV_Test_af_onset.append(Test_af_onset)
    CV_Test_bf_peak.append(Test_bf_peak)
    CV_Test_start_time.append(Test_start_time)
    CV_Test_model_variance.append(Test_model_variance)

    CV_Test_fold.append(
        np.full(len(Test_targets), test_fold[0])
    )

All_Test_targets = np.concatenate(CV_Test_targets, axis=0)
All_Test_results = np.concatenate(CV_Test_results, axis=0)
All_Test_classes = np.concatenate(CV_Test_classes, axis=0)

All_Test_event_id = np.concatenate(CV_Test_event_id, axis=0)
All_Test_af_onset = np.concatenate(CV_Test_af_onset, axis=0)
All_Test_bf_peak = np.concatenate(CV_Test_bf_peak, axis=0)
All_Test_fold = np.concatenate(CV_Test_fold, axis=0)

All_Test_start_time = np.concatenate(CV_Test_start_time, axis=0)
All_Test_model_variance = np.concatenate(CV_Test_model_variance, axis=0)

All_Test_risetime = All_Test_af_onset + All_Test_bf_peak

All_Test_classes = np.where(All_Test_classes == 1, "C",
                            np.where(All_Test_classes == 2, "M",
                                     np.where(All_Test_classes == 3, "X", 
                                              All_Test_classes)))

df_save = pd.DataFrame({
    "event_id": All_Test_event_id,
    "fold": All_Test_fold,
    "start_time": All_Test_start_time,
    "class": All_Test_classes,
    "rise_time": All_Test_risetime,
    "af_onset": All_Test_af_onset,
    "bf_peak": All_Test_bf_peak,
    "target": All_Test_targets,
    "model_result": All_Test_results,
    "model_variance": All_Test_model_variance,
})

df_save = df_save.sort_values(
    by=["event_id", "af_onset"],
    ascending=[True, True]
).reset_index(drop=True)

df_save.to_csv(save_path, index=False)
