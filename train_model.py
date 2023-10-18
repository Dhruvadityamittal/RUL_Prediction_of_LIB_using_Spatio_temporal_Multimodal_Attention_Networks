import shutil
import os
import torch
from torchmetrics.classification import BinaryAccuracy
from util_FPC import EarlyStopping, weight_reset
import torch.nn as nn
from model import CNN_Model, LSTM_Model_RUL, CNN_Model_RUL, Net, Net_new
import time
import numpy as np
from dataloader import get_RUL_dataloader
from torchmetrics import MeanAbsolutePercentageError
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, optimizer, criterion, early_stopping,train_dataloader,epochs,lr, load_pretrained, path,version):
    
    

    metric = BinaryAccuracy().to(device)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        model.requires_grad_(True)
        acc = 0
        total_loss = 0
        total = 0
        total_batches = 0
        for x, y ,_ in train_dataloader:

            x = x.to(device=device)
            y = y.to(device=device)
            out = model(x)
            acc += metric(out, y.unsqueeze(1))

            loss = criterion(out,y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size()[0]
            total += x.size()[0]
            total_batches +=1


        print("Epoch = {} : Loss = {}, Accuarcy = {}".format(epoch, total_loss/total,acc/total_batches))

        evaluation = total_loss/total
        early_stopping(evaluation, model, path)
        
        if early_stopping.early_stop:
            print('Early stopping')
            break
    model.load_state_dict(torch.load(path, map_location=device ))    

    return model




def train_model_RUL(model_RUL,criterion, optimizer,train_dataloader,val_dataloader,epochs,lr,load_pretrained,path,early_stopping,version):
    times = []
    model_RUL.to(device) 
    
    MAE_loss = nn.L1Loss().to(device)
    MSE_loss = nn.MSELoss().to(device)
    MAPE_loss = MeanAbsolutePercentageError().to(device)

    for epoch in range(epochs):
        model_RUL.train()
        model_RUL.requires_grad_(True)
        total_loss = 0
        total_MAE_loss = 0
        total_MSE_loss = 0
        total_MAPE_loss = 0
        total_recon_loss = 0
        total = 0
        total_batches = 0
        start = time.time()
        for x, y ,_ in train_dataloader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            if(model_RUL.name == "Transformer"):
                out,d = model_RUL(x)
                # loss = criterion(out,y.unsqueeze(1))  + 0.01*criterion(d,x)
                loss_MAE = MAE_loss(out,y.unsqueeze(1))
                loss_MSE = MSE_loss(out,y.unsqueeze(1))
                loss_MAPE = MAPE_loss(out,y.unsqueeze(1)) 
                loss_recon = criterion(d,x)
                loss =  loss_MAE + loss_MSE + loss_MAPE  + 0.1* loss_recon
            else:
                out =  model_RUL(x)
                # loss = criterion(out,y.unsqueeze(1))
                loss_MAE = MAE_loss(out,y.unsqueeze(1))
                loss_MSE = MSE_loss(out,y.unsqueeze(1))
                loss_MAPE = MAPE_loss(out,y.unsqueeze(1)) 
                # loss_recon = criterion(d,x)
                loss =  loss_MAE + loss_MSE + loss_MAPE 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size()[0]
            total_MAE_loss += loss_MAE.item() * x.size()[0]
            total_MSE_loss += loss_MSE.item() * x.size()[0]
            total_MAPE_loss += loss_MAPE.item() * x.size()[0]
            if(model_RUL.name == "Transformer"):
                total_recon_loss += loss_recon.item() * x.size()[0]
            total += x.size()[0]
            total_batches +=1

        end = time.time()
        times.append(end-start)
        val_mse, val_mae, val_mape = test_model_RUL(model_RUL, criterion, path, val_dataloader, show_output= False, ValidorTest='valid')
        if(model_RUL.name == "Transformer"):
            print("Epoch = {}, Train Loss = {}, Train_MAE={}, Train_MSE={}, Train_MAPE={}, Train_recon={}, Val MSE = {}, MAE = {}, MAPE = {}".format(epoch+1, total_loss/total, total_MAE_loss/total, total_MSE_loss/total, total_MAPE_loss/total, total_recon_loss/total, val_mse, val_mae, val_mape))

        else:
            print("Epoch = {}, Train Loss = {}, Train_MAE={}, Train_MSE={}, Train_MAPE={}, Val MSE = {}, MAE = {}, MAPE = {}".format(epoch+1, total_loss/total, total_MAE_loss/total, total_MSE_loss/total, total_MAPE_loss/total, val_mse, val_mae, val_mape))
        
        evaluation = val_mse + val_mae + val_mape 
        early_stopping(evaluation, model_RUL,path)

        if early_stopping.early_stop:
            print('Early stopping')
            break
    print("\n Average Time per Epoch :" ,np.mean(times))       
    model_RUL.load_state_dict(torch.load(path, map_location=device ))  
    return model_RUL


def test_model_RUL(model_RUL, criterion, path, test_dataloader, show_output, ValidorTest = 'valid'):
    if ValidorTest == 'test':
        model_RUL.load_state_dict(torch.load(path, map_location=device ))  

    model_RUL.eval()
    model_RUL.requires_grad_(False)
    
    total_loss_mse = 0
    total_mse = 0
    total_loss_mae = 0
    total_mae = 0
    total_loss_mape = 0
    total_mape = 0
    
    total_batches = 0

    

    l1 = nn.L1Loss().to(device)
    l2 = MeanAbsolutePercentageError().to(device)
    for x, y ,_ in test_dataloader:
        x = x.to(device=device)
        y = y.to(device=device)
        
        if(model_RUL.name == "Transformer"):
            out,d = model_RUL(x)
            loss_mse = criterion(out,y.unsqueeze(1))  + 0*criterion(d,x)
            loss_mae = l1(out,y.unsqueeze(1))  + 0*l1(d,x)
            loss_mape = l2(out,y.unsqueeze(1))  + 0*l2(d,x)

        else:
            out =  model_RUL(x)
            loss_mse = criterion(out,y.unsqueeze(1))
            loss_mae = l1(out,y.unsqueeze(1))  
            loss_mape = l2(out,y.unsqueeze(1))


        total_loss_mse += loss_mse.item() * x.size()[0]
        total_mse += x.size()[0]

        total_loss_mae += loss_mae.item() * x.size()[0]
        total_mae += x.size()[0]

        total_loss_mape += loss_mape.item() * x.size()[0]
        total_mape += x.size()[0]

        total_batches +=1

    if(show_output):
        print("\n\nTest loss : MSE = {}, MAE = {}, MAPE = {} \n\n".format(total_loss_mse/total_mse, 
                                                                      total_loss_mae/total_mae,
                                                                        total_loss_mape/total_mape))
    
    return total_loss_mse/total_mse, total_loss_mae/total_mae, total_loss_mape/total_mape
            

def perform_n_folds(model, n_folds,discharge_capacities,change_indices,criterion, 
                    optimizer, early_stopping, load_pretrained, path,
                     scenario, parameters,version,dataset):
    all_MSE =[]
    all_MAE =[]
    all_MAPE =[]
    for fold in range(n_folds):
        print("*********************  Fold = {}  ********************* \n\n".format(fold))
        test_batteries = [i for i in range(len(discharge_capacities)) if i % n_folds == fold]
        train_batteries = [i for i in range(len(discharge_capacities)) if i not in test_batteries]
        val_batteries = random.sample(train_batteries,int(0.1*len(train_batteries)))
        train_batteries = [i for i in range(len(discharge_capacities)) if i not in val_batteries]
        
        fold_path = path[:-10]+"_Fold"+str(fold)+".pth"
    
        train_dataloader_RUL, train_dataloader_RUL_temp, val_dataloder_RUL, test_dataloader_RUL = get_RUL_dataloader(discharge_capacities, 
                                                                                              train_batteries, val_batteries, test_batteries, 
                                                                                              change_indices, parameters["window_size"],
                                                                                                parameters["stride"],parameters["channels"] ,scenario)
        
        early_stopping = EarlyStopping(patience=20)
        model = train_model_RUL(model, criterion, optimizer,train_dataloader_RUL,val_dataloder_RUL,parameters["epochs"],
                                parameters["learning_rate"],load_pretrained,fold_path,early_stopping,version)
        
        mse, mae, mape = test_model_RUL(model,criterion, fold_path, test_dataloader_RUL, show_output=True, ValidorTest='test')
        all_MSE.append(mse)
        all_MAE.append(mae)
        all_MAPE.append(mape)

        np.save(f"./Test_data/test_batteries_{dataset}_{model.name}_fold{fold}.npy", test_batteries, allow_pickle=True)
        np.save(f"./Test_data/train_batteries_{dataset}_{model.name}_fold{fold}.npy", train_batteries, allow_pickle=True)
        np.save(f"./Test_data/val_batteries_{dataset}_{model.name}_fold{fold}.npy", train_batteries, allow_pickle=True)
        if(fold !=n_folds-1):
            model.apply(weight_reset)
        else:
            print("Average Folds MSE = {} , MAE = {}, MAPE = {}  ".format(np.mean(all_MSE),np.mean(all_MAE),np.mean(all_MAPE)))    
            print("STD Folds MSE = {} , MAE = {}, MAPE = {}  ".format(np.std(all_MSE),np.std(all_MAE),np.std(all_MAPE)))    
            return model, test_dataloader_RUL, test_batteries, train_batteries, val_batteries
    
        

            
            

