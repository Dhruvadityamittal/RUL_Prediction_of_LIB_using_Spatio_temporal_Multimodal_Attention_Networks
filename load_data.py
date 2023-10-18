import numpy as np
import torch
import os 

def get_dirs():
    important_dirs = ["Weights","Weights/FPC","Weights/Scenario1","Weights/Scenario2","Outputs"]

    for dir in important_dirs:
        if(os.path.isdir(dir)==False):
            print("Creating Directory :", dir)
            os.mkdir(dir)

def get_discharge_capacities_MIT():
      discharge_capacities = np.load(r"Datasets/discharge_capacity.npy", allow_pickle=True)
      return  discharge_capacities.tolist()

def get_discharge_capacities_HUST(fea_num):
    discharge_capacities = np.load(f"./Datasets/snl_data_{fea_num}.npy",allow_pickle=True)
    d = []
    for battery_temp in discharge_capacities:
        a = np.squeeze(battery_temp[0], axis = 1)     # Voltage/Current Features
        b = np.expand_dims(battery_temp[1], axis =1)  # Discharge Capccity
        c = np.concatenate((b,a), axis =1).T
        d.append(c)
    return d

def get_data(discharge_capacities,percentage,window_size,stride,channels,type, name_start):

    train_data =[]
    FPC_data  =[]
    name = name_start
    test_data = []
    FPC_data_dict ={}
    test_data_dict = {}
    if(type == "train"):
        
        for battery in discharge_capacities:
            a = len(FPC_data)
#             battery = np.asarray(battery)
            
            battery = np.asarray([battery[i] for i in channels])
            battery_name = 'battery' + str(name)
            FPC_data_dict[battery_name] =[]
            name = name+1
            
            # Taking inital x% as input and giving the output as 1
            i= 0
            target = 1
            while(i+stride+window_size <= int(percentage*len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                train_data.append((battery[:,i:i+window_size], target,battery_name ))
                i = i+stride

            # Taking inputs in the middle for FPC
            i = int(percentage*len(battery[0]))
            target = -1
            while(i+stride+window_size <= int((1-percentage)*len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                FPC_data.append((battery[:,i:i+window_size], target,battery_name))
                FPC_data_dict[battery_name].append(torch.tensor(battery[:,i:i+window_size]).float())
                i = i+stride

            # Taking last x% as input and giving the output as 0
            i = int((1-percentage)*len(battery[0]))
            target = 0
            while(i+stride <= len(battery[0]) and len(battery[0][i:i+window_size]) == window_size):
                train_data.append((battery[:,i:i+window_size], target ,battery_name))
                i = i+stride
            # print(len(FPC_data)-a, len(battery[0]), len(FPC_data)-a- .90*len(battery[0]))

        return train_data,FPC_data,FPC_data_dict

    else:
        name = name_start
        for battery in discharge_capacities:
            
            battery = np.asarray([battery[i] for i in channels])
            i= 0
            battery_name = 'battery' + str(name)
            test_data_dict[battery_name] =[]
            name = name+1
            while(i+stride <= len(battery[0]) and len(battery[0][i:i+window_size]) == window_size):
                test_data.append((battery[:,i:i+window_size], 1,battery_name))
                test_data_dict[battery_name].append(torch.tensor(battery[:,i:i+window_size]).float())
                i = i+stride

        return test_data,test_data_dict

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)), (max(data), min(data))

def get_data_RUL_scenario1(discharge_capacities,batteries, change_indices, window_size,stride,channels):
        
        data =[]
        for bat in (batteries):
                battery = np.asarray([discharge_capacities[bat][i] for i in channels])
                battery_name = "battery" + str(bat)
                i = change_indices[bat]   # FPC cycle
                
                percentage_index = 0
                
                EOL = len(battery[0])

                while(i+stride+window_size+1 <= int(len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                        data.append((battery[:,i:i+window_size], 1-((i-change_indices[bat])/(EOL - change_indices[bat])),battery_name ))
                        i = i+stride
                        percentage_index = percentage_index+1

        return data


def get_data_RUL_scenario2(discharge_capacities,change_indices, window_size,stride,channels, type):
                
        data =[]
        for bat in (discharge_capacities):
                battery = np.asarray([battery[bat][i] for i in channels])
                battery_name = "battery" + str(bat)
                i = change_indices[bat]
                
                percentage_index = 0
                normalized_capacity,_ = NormalizeData(battery[0][i:])

                while(i+stride+window_size+1 <= int(len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                        data.append((battery[:,i:i+window_size], normalized_capacity[percentage_index],battery_name ))
                        i = i+stride
                        percentage_index = percentage_index+1

        return data
