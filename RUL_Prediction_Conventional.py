import numpy as np
import torch
import torch.nn as nn
import pandas

from matplotlib.pylab import plt
import warnings
from load_data_conventional import *
from dataloader_conventional import *
from train_model_conventional import *
from model_conventional import *
from load_data import get_discharge_capacities_HUST,get_discharge_capacities_MIT
import argparse

warnings.filterwarnings('ignore')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    parser.add_argument("--dataset", type=str, default='HUST', help="MIT or HUST")
    parser.add_argument("--m_name", type=str, default='LSTM', help="CNN or LSTM")
    parser.add_argument("--percentage", type=float, default='0.40', help="Percentage of input")
    parser.add_argument("--epochs", type=int, default='500', help="Epochs")

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Selecting the Dataset
    # MIT Dataset
    if(args.dataset == "MIT"):
        discharge_capacities = np.load(r"./Datasets/discharge_capacity.npy", allow_pickle=True)
        discharge_capacities = discharge_capacities.tolist()
        channels = [0]

    # HUST Dataset
    elif(args.dataset == "HUST"):    
        discharge_capacities = get_discharge_capacities_HUST(fea_num=1)
        channels = [0]
    
    # Combined Dataset
    else:
        discharge_capacities_MIT = get_discharge_capacities_MIT()
        discharge_capacities_HUST = get_discharge_capacities_HUST(fea_num=1)
        discharge_capacities = discharge_capacities_MIT[0:100] + discharge_capacities_HUST[0:70] + discharge_capacities_MIT[100:] + discharge_capacities_HUST[70:]
        channels = [0]



    # Getting the maximum number of cycles in the dataset
    max_length_train,max_length_out = get_lengths(discharge_capacities,args.percentage)
    max_length_out = max_length_out+1

    # Selecting the model.
    if(args.m_name == "LSTM"):
        model = LSTM_Model_Conventional(max_length_train,len(channels),max_length_out)
    else:
        model = CNN_Model_Conventional(max_length_train,len(channels),max_length_out)


    ch = ''.join(map(str,channels))
    version = 1
    fld = 3

    print(device,model.name, args.dataset)

    model_dir = "./Weights/Conventional/"
    model_path = f'{model_dir}/{args.dataset}_{model.name}_Conventional_Channels={ch}_Version={version}_Fold{fld}.pth'
   
    load_pretrained = False
    pretrained = False
    lr = 0.001
    n_folds = 5

    parameters = {"epochs" : args.epochs,
                    "learning_rate" : lr ,
                    "percentage" : args.percentage,
                    "max_length_train" :max_length_train,
                    "max_length_out" :max_length_out,
                    "channels" : channels
    }

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas= (0.9, 0.99))
    criterion = nn.MSELoss()
    early_stopping_patiance = 50

    if(pretrained):
        model.load_state_dict(torch.load(model_path,map_location= device))
    else:
        if(load_pretrained):
            model.load_state_dict(torch.load(model_path,map_location= device))

            model= perform_n_folds_conventional(model, n_folds,discharge_capacities, criterion, 
                            optimizer, early_stopping_patiance, model_path,
                            parameters,version, args.dataset)
        else:
            model = perform_n_folds_conventional(model, n_folds,discharge_capacities, criterion, 
                        optimizer, early_stopping_patiance, model_path,
                        parameters,version, args.dataset)