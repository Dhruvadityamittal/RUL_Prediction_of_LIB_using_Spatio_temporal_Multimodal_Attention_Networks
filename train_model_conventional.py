from torchmetrics import MeanAbsolutePercentageError
import random
from util_FPC import weight_reset,EarlyStopping
from import_file import *
from dataloader_conventional import get_conventional_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import time
def train_model_Conventaional(model,criterion, optimizer,train_dataloader,val_dataloader,epochs,path,early_stopping):
    times = []
    model.to(device) 
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        model.requires_grad_(True)
        total_loss = 0
        total = 0
        total_batches = 0
        start = time.time()
        for x, y  in train_dataloader:
            
            x = x.to(device=device)
            y = y.to(device=device)
            
            if(model.name == "Transformer"):
                out,d = model(x)
                loss = criterion(out,y.unsqueeze(1))  + 0*criterion(d,x)
            else:
                out =  model(x)
                loss = criterion(out,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size()[0]
            total += x.size()[0]
            total_batches +=1

        end = time.time()
        times.append(end-start)
        val_mse, val_mae, val_mape,_ = test_model_conventional(model, criterion, val_dataloader, show_output= False)
        
        print("Epoch = {}, Train Loss = {}, Val MSE = {}, MAE = {}, MAPE = {}".format(epoch, total_loss/total,val_mse, val_mae, val_mape))
        
        evaluation = 0.75*val_mse + 0.25*val_mae 
        # + 0.0020*val_mape
        # evaluation = 0.75*val_mse + 0.15*val_mae + 0.20*val_mape 
        early_stopping(evaluation, model,path)

        if early_stopping.early_stop:
            print('Early stopping')
            break
    print("\n Average Time per Epoch :" ,np.mean(times))       
    model.load_state_dict(torch.load(path, map_location=device ))  
    return model

def loss_fn(output, target):
    # MAPE loss

    return torch.mean(torch.abs((target - output) / (target+0.01))).to(device)

def test_model_conventional(model, criterion, test_dataloader, show_output):

    total_loss_mse = 0
    total_mse = 0
    total_loss_mae = 0
    total_mae = 0
    total_loss_mape = 0
    total_mape = 0

    total_batches = 0
    outputs = []
    l1 = nn.L1Loss().to(device)
    # l2 = MeanAbsolutePercentageError().to(device)

    for x, y in test_dataloader:

        x = x.to(device=device)
        y = y.to(device=device)
        
        # print(x.shape, list(x[0][0]).index(-0.1))
        # print(len(y[0]),list(y[0]).index(-0.1))
        
        out =  model(x)
        
        padding_start = list(y[0]).index(-0.1)
        padding_removed_y = y[0][:padding_start]
        padding_removed_out = out[0][:padding_start]

        outputs.append((y[0][:padding_start],out[0][:padding_start]))
        
        
        loss_mse = criterion(padding_removed_out,padding_removed_y)
        loss_mae = l1(padding_removed_out,padding_removed_y)  
        loss_mape = loss_fn(padding_removed_out,padding_removed_y)
        # l2(padding_removed_out,padding_removed_y)


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
    
    return total_loss_mse/total_mse, total_loss_mae/total_mae, total_loss_mape/total_mape, outputs
    


def perform_n_folds_conventional(model, n_folds,discharge_capacities,criterion, 
                    optimizer, early_stopping_patiance, path,
                    parameters,version, dataset):
    
    for fold in range(n_folds):
        print("*********************  Fold = {}  ********************* \n\n".format(fold))
        test_batteries = [i for i in range(len(discharge_capacities)) if i % n_folds == fold]
        train_batteries = [i for i in range(len(discharge_capacities)) if i not in test_batteries]
        val_batteries = random.sample(train_batteries,int(0.1*len(train_batteries)))
        train_batteries = [i for i in range(len(discharge_capacities)) if i not in val_batteries]
        
        
        fold_path = path[:-10]+"_Fold"+str(fold)+".pth"
    
        train_dataloader, train_dataloader_RUL_temp, val_dataloder, test_dataloader = get_conventional_dataloader(discharge_capacities,train_batteries,
                                                                                                                  test_batteries,val_batteries,parameters["channels"],
                                                                                                                  parameters["max_length_train"],
                                                                                                                  parameters["max_length_out"],parameters["percentage"])
        
        early_stopping = EarlyStopping(patience=early_stopping_patiance)
        model = train_model_Conventaional(model, criterion, optimizer,train_dataloader,val_dataloder,parameters["epochs"],
                                fold_path,early_stopping)
        
        test_model_conventional(model,criterion,test_dataloader,show_output=True)
        np.save(f"./Test_data/test_batteries_Conventional_{dataset}_{model.name}_fold{fold}_version{version}.npy", test_batteries, allow_pickle=True)
        np.save(f"./Test_data/train_batteries_Conventional_{dataset}_{model.name}_fold{fold}_version{version}.npy", train_batteries, allow_pickle=True)
        np.save(f"./Test_data/val_batteries_Conventional_{dataset}_{model.name}_fold{fold}_version{version}.npy", val_batteries, allow_pickle=True)
        if(fold !=n_folds-1):
            model.apply(weight_reset)
        else:
            
            return model
        # , test_dataloader, test_batteries, train_batteries, val_batteries