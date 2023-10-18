import torch
from torch.utils.data import Dataset, DataLoader
from load_data import get_data_RUL_scenario1, get_data_RUL_scenario2



class battery_dataloader(Dataset):
    
    def __init__(self,data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        inp =  torch.tensor(self.data[idx][0]).float()
        output = torch.tensor(self.data[idx][1]).float()
        name = self.data[idx][2]
        return inp, output,name 

class battery_dataloader_RUL(Dataset):
    
    def __init__(self,data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        inp =  torch.tensor(self.data[idx][0]).float()
        output = torch.tensor(self.data[idx][1]).float()
        battery_name = self.data[idx][2]
        
        return inp, output, battery_name
    
def get_RUL_dataloader(discharge_capacities, train_batteries, val_batteries,test_batteries, change_indices, window_size, stride, channels,scenario):
    
    if(scenario ==1):
       
        train_data_RUL_scenario1= get_data_RUL_scenario1(discharge_capacities,train_batteries,change_indices,window_size,stride,channels)
        obj_train_RUL_scenario1  = battery_dataloader_RUL(train_data_RUL_scenario1)

        val_data_RUL_scenario1= get_data_RUL_scenario1(discharge_capacities,val_batteries,change_indices,window_size,stride,channels)
        obj_val_RUL_scenario1  = battery_dataloader_RUL(val_data_RUL_scenario1)

        test_data_RUL_scenario1= get_data_RUL_scenario1(discharge_capacities,test_batteries,change_indices,window_size,stride,channels)
        obj_test_RUL_scenario1  = battery_dataloader_RUL(test_data_RUL_scenario1)


        train_dataloader_RUL_scenario1 = DataLoader(obj_train_RUL_scenario1, batch_size=128,shuffle=True)
        train_dataloader_RUL_temp_scenario1 = DataLoader(obj_train_RUL_scenario1, batch_size=1,shuffle=False)
        test_dataloader_RUL_scenario1 = DataLoader(obj_test_RUL_scenario1, batch_size=1,shuffle=False)
        val_dataloader_RUL_scenario1 = DataLoader(obj_val_RUL_scenario1, batch_size=1,shuffle=False)

        return train_dataloader_RUL_scenario1, train_dataloader_RUL_temp_scenario1, val_dataloader_RUL_scenario1, test_dataloader_RUL_scenario1

    else:
        train_data_RUL_scenario2= get_data_RUL_scenario2(discharge_capacities,train_batteries,change_indices,window_size,stride,channels)
        obj_train_RUL_scenario2  = battery_dataloader_RUL(train_data_RUL_scenario2)

        val_data_RUL_scenario2= get_data_RUL_scenario2(discharge_capacities,val_batteries,change_indices,window_size,stride,channels)
        obj_val_RUL_scenario2  = battery_dataloader_RUL(val_data_RUL_scenario2)

        test_data_RUL_scenario2= get_data_RUL_scenario2(discharge_capacities,test_batteries,change_indices,window_size,stride,channels)
        obj_test_RUL_scenario2  = battery_dataloader_RUL(test_data_RUL_scenario2)

        train_dataloader_RUL_scenario2 = DataLoader(obj_train_RUL_scenario2, batch_size=128,shuffle=True)
        train_dataloader_RUL_temp_scenario2 = DataLoader(obj_train_RUL_scenario2, batch_size=1,shuffle=False)
        test_dataloader_RUL_scenario2 = DataLoader(obj_test_RUL_scenario2, batch_size=1,shuffle=False)
        val_dataloader_RUL_scenario2 = DataLoader(obj_val_RUL_scenario2, batch_size=1,shuffle=False)

        return train_dataloader_RUL_scenario2, train_dataloader_RUL_temp_scenario2,val_dataloader_RUL_scenario2, test_dataloader_RUL_scenario2