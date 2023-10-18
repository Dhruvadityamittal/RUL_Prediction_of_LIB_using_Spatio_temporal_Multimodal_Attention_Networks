from load_data_conventional import get_data_conventional
from torch.utils.data import Dataset, DataLoader
import torch 


def get_conventional_dataloader(discharge_capacities,train_batteries,test_batteries,val_batteries,channels,max_length_train,max_length_out,percentage):
    
    data_train = get_data_conventional(discharge_capacities, channels,train_batteries,max_length_train, max_length_out,percentage)
    data_test =  get_data_conventional(discharge_capacities, channels,test_batteries,max_length_train, max_length_out,percentage)
    val_batteries = get_data_conventional(discharge_capacities,channels, val_batteries,max_length_train, max_length_out,percentage)

    obj_train  = battery_dataloader_conventional(data_train)
    obj_test  = battery_dataloader_conventional(data_test)
    obj_val  = battery_dataloader_conventional(val_batteries)

    train_dataloader = DataLoader(obj_train, batch_size=8,shuffle=True)
    train_dataloader_temp = DataLoader(obj_train, batch_size=1,shuffle=False)
    test_dataloader = DataLoader(obj_test, batch_size=1,shuffle=False)
    val_dataloader = DataLoader(obj_val, batch_size=1,shuffle=False)

    return train_dataloader,train_dataloader_temp,val_dataloader , test_dataloader

class battery_dataloader_conventional(Dataset):
    
    def __init__(self,data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        
        inp =  torch.tensor(self.data[idx][0]).float()
        output = torch.tensor(self.data[idx][1]).float()
        
        return inp, output