import numpy as np

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)), (max(data), min(data))

def get_lengths(discharge_capacities,percentage):
    lengths = [len(battery[0]) for battery in discharge_capacities]
    max_length = max(lengths)   # Finding maximum cycle length

    cell_with_max_cycle_life = np.argmax(lengths)

    print("Cell with Maximum length :", cell_with_max_cycle_life)
    print("Maximum cycle length :", max_length)        # Percentage of data for training
                                                  
    max_length_train = int(percentage*max_length)                         # maximum length for training
    max_length_out  = max_length- max_length_train                        # maximum length of the output

    print("Maximum Length for training X:", max_length_train)
    print("Maximum Length of output     :", max_length_out)

    return max_length_train, max_length_out


def get_data_conventional(discharge_capacities,channels, batteries,max_length_train,max_length_out, percentage):
    
    data = []
    for battery_no in batteries:

        battery = discharge_capacities[battery_no]   
        total_number_of_cycles = len(battery[0])
        number_of_cycles_train = int(total_number_of_cycles*percentage)   # getting some percentage of data for the training 
        number_of_cycles_out  = total_number_of_cycles - number_of_cycles_train  # remaining data is the output

        battery = np.asarray([battery[i] for i in channels])
        # right padding is done in both input and outputs

        xnpad = [(0, 0)] * 2
        xnpad[1] = (0,max_length_train-number_of_cycles_train)
        
        X =   np.pad(battery[:,:number_of_cycles_train], pad_width=xnpad,
                                    mode = 'constant', constant_values=(-0.1))
        
        # index zero contains discharge Capacity
        Y = np.pad(NormalizeData(battery[0,number_of_cycles_train:])[0], (0,max_length_out-number_of_cycles_out),
                                    mode = 'constant', constant_values=(-0.1))

        # Y_train_padded.append(np.pad(NormalizeData(discharge_capacities[i][number_of_cycles_train:])[0], (0,max_length_out-number_of_cycles_out),'constant', constant_values=(-0.1)))
    
        data.append((X,Y))

    return data