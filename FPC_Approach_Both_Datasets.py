import os
import argparse

os.chdir(".")

from model import CNN_Model, LSTM_Model_RUL, CNN_Model_RUL, Net, Net_new, Autoencoder, LSTM_Model, TransformerLSTM
from load_data import get_data, get_data_RUL_scenario1, get_discharge_capacities_MIT,get_discharge_capacities_HUST, get_dirs, get_data_RUL_scenario2
from dataloader import battery_dataloader, battery_dataloader_RUL, get_RUL_dataloader
from import_file import *
from train_model import train_model, train_model_RUL, test_model_RUL, perform_n_folds
from util_FPC import get_fpc_window, get_data, get_fpc, get_change_indices, EarlyStopping, plot_RUL, weight_reset
parser = argparse.ArgumentParser()
    

if __name__ == "__main__":
    get_dirs()  # Make all the necessary libraries.. 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nTraining on ", device)

    # Arguments
    parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    parser.add_argument("--dataset", type=str, default='HUST', help="MIT or HUST")
    parser.add_argument("--RUL_model_name", type=str, default='Transformer', help="LSTM, Net, CNN or Transformer(Proposed)")
    parser.add_argument("--percentage", type=float, default='0.10', help="Percentage for Heath Stage Prediction")
    parser.add_argument("--window_size_FPC", type=int, default='50', help="Window Size for Health Stage Prediction")
    parser.add_argument("--stride_FPC", type=int, default='1', help="Stride for Health Stage Prediction")
    parser.add_argument("--window_size_RUL", type=int, default='64', help="Window for RUL Prediction")
    parser.add_argument("--stride_RUL", type=int, default='1', help="Stride for RUL Prediction")
    parser.add_argument("--epochs_FPC", type=int, default='50', help="Epochs for Health Stage Prediction")
    parser.add_argument("--epochs_RUL", type=int, default='600', help="Epochs for RUL Prediction")
    args = parser.parse_args()
    
    print("Using Dataset :", args.dataset)

    if(args.dataset == "MIT"):
        channels  =[0,1,2,3,4,5,6]                        #channels used for HS prediction
        channels_RUL = [0,1,2,3,4,5,6]                    #channels used for RUL prediction

        name_start_train = 0                              #Battery number to start from
        name_start_test = 100                             #Battery number to end
        
        discharge_capacities = get_discharge_capacities_MIT()            # Get Discharge Capacity Data
        learning_rate_FPC = 0.0001
        
        model_FPC = LSTM_Model(args.window_size_FPC,len(channels))            # Select model for Health Stage Prediction

        if(args.RUL_model_name == "LSTM"):
            model_RUL = LSTM_Model_RUL(args.window_size_RUL,len(channels_RUL))  # LSTM Model
            learning_rate_RUL = 0.0001
        elif(args.RUL_model_name == "Net"):
            model_RUL = Net(len(channels_RUL), feature_size=args.window_size_RUL)    # Transformer Model
            learning_rate_RUL = 0.0001      
        elif(args.RUL_model_name == "Transformer"):
            model_RUL = TransformerLSTM(len(channels_RUL), args.window_size_RUL)
            learning_rate_RUL = 0.0001
        else:
            model_RUL = CNN_Model_RUL(args.window_size_RUL,len(channels_RUL))  # CNN Model
            learning_rate_RUL = 0.0001      
        
    else:                               # For HUST dataset
        channels  =[0]                  # channels used for Health Stage Prediction
        channels_RUL = [0,1,2,3,4]
        name_start_train = 0               #Battery number to start from
        name_start_test = 70               #Battery number to end

        discharge_capacities = get_discharge_capacities_HUST(fea_num =1)
        learning_rate_FPC = 0.0001
        # model_FPC = CNN_Model(window_size,len(channels))
        model_FPC = LSTM_Model(args.window_size_FPC,len(channels))

        if(args.RUL_model_name == "LSTM"):
            model_RUL = LSTM_Model_RUL(args.window_size_RUL,len(channels_RUL))  # LSTM Model
            learning_rate_RUL = 0.0001
        elif(args.RUL_model_name == "Net"):
            model_RUL = Net(len(channels_RUL))    # Transformer Model
            learning_rate_RUL = 0.001      # CNN Model
        elif(args.RUL_model_name == "Transformer"):
            model_RUL = TransformerLSTM(len(channels_RUL), args.window_size_RUL)
            learning_rate_RUL = 0.001
        else:
            model_RUL = CNN_Model_RUL(args.window_size_RUL,len(channels_RUL))
            learning_rate_RUL = 0.0001      # CNN Model


    
    """************************ Training for Stage 1   ************************"""

    print("\n***************** Stage 1 : Health Stage Prediction ********************\n")

    # Getting data for training and Helth Stage Calculation from initial batteries
    train_data,FPC_data,FPC_data_dict = get_data(discharge_capacities[:name_start_test],args.percentage,
                                                 args.window_size_FPC,args.stride_FPC,channels,type = "train",
                                                   name_start=name_start_train)

    # Getting data for testing in HS Stage Calculation
    test_data,test_data_dict  = get_data(discharge_capacities[name_start_test:],None,args.window_size_FPC,
                                         args.stride_FPC,channels,type= "test",
                                           name_start=name_start_test)

    # Generating the dataloader objects
    obj_train  = battery_dataloader(train_data)
    obj_FPC  = battery_dataloader(FPC_data)
    obj_test  = battery_dataloader(test_data)

    # Getting dataloaders
    train_dataloader = DataLoader(obj_train, batch_size=8,shuffle=True)
    FPC_dataloader   = DataLoader(obj_FPC,batch_size=1,shuffle=False)
    test_dataloader = DataLoader(obj_test, batch_size=1,shuffle=False)

    print("Using Channels  :", channels)
    print("Shape of a batch    :",next(iter(train_dataloader))[0].shape)


    pretrained = True
    load_pretrained = False
    print(r"Using a Pretrained Model :{}, loading Pretrained model to train further :{}".format(pretrained,load_pretrained))

    version = 1
    print("Using Version :", version)

    ch = ''.join(map(str,channels))
    model_dir = "./Weights/FPC/"
    model_path = f'{model_dir}/{args.dataset}_{model_FPC.name}_FPC_Channels={ch}_WindowSize={args.window_size_FPC}_Version={version}.pth'
    
    model_FPC.to(device)
    optimizer = torch.optim.Adam(model_FPC.parameters(), lr = learning_rate_FPC, betas= (0.9, 0.99))
    criterion = nn.BCELoss()
    early_stopping = EarlyStopping(patience=20)

    if(pretrained):    
        model_FPC.load_state_dict(torch.load(model_path, map_location=device ))
        model_FPC.to(device)
        print(r"Loading Pretrained Model from :{}".format(model_path))
    else:
        if(load_pretrained):
            print(r"Training further on already trained model from :{}".format(model_path))
            model_FPC.load_state_dict(torch.load(model_path, map_location=device ))
            model_FPC = train_model(model_FPC, optimizer, criterion, early_stopping,train_dataloader,args.epochs_FPC,learning_rate_FPC,load_pretrained,model_path,version)
            
        else:
            print(r"Training the model from Scratch again")
            model_FPC = train_model(model_FPC, optimizer, criterion, early_stopping,train_dataloader,args.epochs_FPC,learning_rate_FPC,load_pretrained,model_path,version)

    # Get the point where Health Stage changes from Healthy to Unhealthy (FPC Stage)
    change_indices_train,change_indices_test, _, _ = get_change_indices(model_FPC,discharge_capacities,channels,get_saved_indices = True, version = 1, name_start_train = name_start_train,name_start_test= name_start_test , dataset= args.dataset) 
    change_indices_all = np.concatenate((change_indices_train,change_indices_test))

    # Inference on Test and Train

    # batteries = [i for i in range(0,24)]    # batteries to infer
    # _,_ = get_fpc(model_FPC,batteries,discharge_capacities,FPC_data_dict,True, True,True,"./Outputs/FPC_Training_" + dataset + "_train_new_latest")

    # batteries = [i+name_start_test for i in range(0,len(discharge_capacities)-name_start_test)]
    # _,_= get_fpc(model_FPC,batteries,discharge_capacities,test_data_dict,True, False,False,"Outputs/FPC_Testing_" + dataset + "_test_new_latest")

    # **************************************************************************
    # RUL prediction code Starts here



    """************************ Training for Stage 2   ************************"""

    print("\n\n***************** Stage 2 : RUL Prediction Stage ********************\n")

    c_RUL = ''.join(map(str,channels_RUL))
    n_folds = 5
    scenario = 1

    # Difining parameters
    parameters = {
        "window_size" : args.window_size_RUL,
        "stride": args.stride_RUL,
        "channels": channels_RUL,
        "epochs": args.epochs_RUL,
        "learning_rate": learning_rate_RUL
    }

    print("\n*****************Stage 2 : RUL Prediction ********************\n")
    print("Learning Rate for RUL Prediction :", learning_rate_RUL)
    print("Using Channels  :", channels_RUL)
    print("Training RUL on :", model_RUL.name)


    # Definign Optimizer and loss functions
    optimizer = torch.optim.Adam(model_RUL.parameters(), lr = learning_rate_RUL, betas= (0.9, 0.99))
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=50)

    version = 1
    print("Using Version :", version)

    pretrained_RUL_scenario1 = True
    load_pretrained_scenario1  = False
    print(r"Using a Pretrained Model :{}, loading Pretrained model to train further :{}".format(pretrained,
                                                                                                load_pretrained))

    
    fld = 1  # Fold for inference
    model_dir_scenario1 = "./Weights/Scenario1"  # Model Weights Directory
    model_path_scenario1 = f'{model_dir_scenario1}/{args.dataset}_f{model_RUL.name}_RUL_Channels={c_RUL}_WindowSize={args.window_size_RUL}_Version={version}__Fold{fld}.pth'

    if(pretrained_RUL_scenario1):
        print("Loading a Pre-trained Model")
        print(r"Fetching Pretrained Model from :{}".format(model_path_scenario1))

        # Loading the Pretrained Model
        model_RUL.load_state_dict(torch.load(model_path_scenario1,map_location= device))

        # Getting training, testing and validation data from the specific fold.
        test_batteries = np.load(f"./Test_data/test_batteries_{args.dataset}_{model_RUL.name}_fold{fld}.npy",allow_pickle=True)
        train_batteries = np.load(f"./Test_data/train_batteries_{args.dataset}_{model_RUL.name}_fold{fld}.npy",allow_pickle=True)
        val_batteries = np.load(f"./Test_data/val_batteries_{args.dataset}_{model_RUL.name}_fold{fld}.npy",allow_pickle=True)
    else:
        if(load_pretrained_scenario1):
            print(r"Training further on already trained model from :{}".format(model_path_scenario1))
            model_RUL.load_state_dict(torch.load(model_path_scenario1,map_location= device))
            model_RUL, test_dataloader, test_batteries, train_batteries, val_batteries = perform_n_folds(model_RUL,n_folds,discharge_capacities,change_indices_all,criterion, optimizer, early_stopping,
                        pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version, args.dataset)
        else:
            print(r"Training the model from Scratch again")
            model_RUL, test_dataloader, test_batteries, train_batteries, val_batteries = perform_n_folds(model_RUL,n_folds,discharge_capacities,change_indices_all,criterion, optimizer, early_stopping,
                        pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version, args.dataset)
          


    # test_batteries  = [i+100 for i in range(24)]

    # Inferenece of the model on the given fold
    _, _,  _, test_dataloader_RUL = get_RUL_dataloader(discharge_capacities, train_batteries,val_batteries, test_batteries,
                                                change_indices_all, parameters["window_size"],
                                                parameters["stride"],parameters["channels"] ,scenario)

    # Plot the results
    plot_RUL(model_RUL,discharge_capacities,test_batteries,test_dataloader_RUL,change_indices_all,"Outputs/scenario1_RUL_prediction_"+args.dataset+ "_"+ args.RUL_model_name +"_test")