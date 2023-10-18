
import torch.nn as nn
import math

# Model for Conventional methods

class LSTM_Model_Conventional(nn.Module):
    
    def __init__(self,input_size,channels,max_len_out):
        super(LSTM_Model_Conventional, self).__init__()
        self.name = "LSTM"
        hidden_size1 = input_size
        hidden_size2 = input_size
        num_layers = 4
        self.LSTM1 = nn.LSTM(input_size = input_size, hidden_size = 64, num_layers = num_layers,batch_first=True)
        self.LSTM2 = nn.LSTM(input_size = 64, hidden_size = 32, num_layers = num_layers,batch_first=True)
        # self.LSTM3 = nn.LSTM(input_size = 32, hidden_size = 16, num_layers = num_layers,batch_first=True)
        # self.LSTM4 = nn.LSTM(input_size = 16, hidden_size = 32, num_layers = num_layers,batch_first=True)

        # self.LSTM5 = nn.LSTM(input_size = 32, hidden_size = 64, num_layers = num_layers,batch_first=True)
        # self.LSTM6 = nn.LSTM(input_size = 16, hidden_size = 32, num_layers = num_layers,batch_first=True)

        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(channels*32,128)   # 224 MIT, 160 - HUST
        self.Linear2 = nn.Linear(128,max_len_out)
        # self.Linear3 = nn.Linear(,1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        
        # self.h0 = torch.randn(4, x.size(0), 100)
        # self.c0 = torch.randn(4, x.size(0), 100)
        
        # x= x.view(x.shape[0],channels,x.shape[1])
        # out, (hn, cn) = self.LSTM(x, (self.h0, self.c0))

        out, (_, _) = self.LSTM1(x)
        out,(_,_) = self.LSTM2(out)
        
        out = self.flatten(out)
        out = self.relu(out)
        out = self.Linear1(out)
        out = self.relu(out)
        out = self.Linear2(out)
        # out = self.relu(out)
        # print(out.shape)
        # out = self.Linear3(out)
        out = self.sigmoid(out)


        return out
    

class CNN_Model_Conventional(nn.Module):
    
    def __init__(self,input_size, channels,max_length_out):
        super(CNN_Model_Conventional, self).__init__()
        self.name = "CNN"
        filter_size_1 = 21
        
        
        self.conv1 = nn.Conv1d(channels,16,kernel_size = filter_size_1, stride=1,padding = filter_size_1//2)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(2)
        

        self.conv2 = nn.Conv1d(16,32, kernel_size = filter_size_1, stride = 1,padding = filter_size_1//2)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32,64, kernel_size = filter_size_1, stride = 1,padding = filter_size_1//2)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64,32, kernel_size = filter_size_1, stride = 1,padding = filter_size_1//2)
        self.batch_norm4 = nn.BatchNorm1d(32)
        self.max_pool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(32,16, kernel_size = filter_size_1, stride = 1,padding = filter_size_1//2)
        self.batch_norm5 = nn.BatchNorm1d(16)
        self.max_pool5 = nn.MaxPool1d(2)

       
        self.flatten_size = 16*math.floor(input_size/(2*2*2*2*2))
        self.flatten = nn.Flatten(start_dim=1)
        
        
        self.Linear1 = nn.Linear(self.flatten_size, 100)
        self.batch_norm_linear = nn.BatchNorm1d(100)
        # self.a = nn.Linear()
        self.Linear2 = nn.Linear(100,max_length_out)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        # print(self.flatten_size)

    def forward(self,x):
        # x= x.view(x.shape[0],1,x.shape[1])
        
        out = self.conv1(x)
        # out = self.relu(out)
        out = self.batch_norm1(out)
        # out = self.dropout(out)
        out = self.max_pool1(out)
        

        out = self.conv2(out)
        # out = self.relu(out)
        out = self.batch_norm2(out)
        # out = self.dropout(out)
        out = self.max_pool2(out)   
        
        out = self.conv3(out)
        # out = self.relu(out)
        out = self.batch_norm3(out)
        # out = self.dropout(out)
        out = self.max_pool3(out)
       
        out = self.conv4(out)
        # out = self.relu(out)
        out = self.batch_norm4(out)
        # out = self.dropout(out)
        out = self.max_pool4(out)
    

        out = self.conv5(out)
        # out = self.relu(out)
        out = self.batch_norm5(out)
        # out = self.dropout(out)
        out = self.max_pool5(out)

        out = self.flatten(out)
        out = self.Linear1(out)  
        out = self.Linear2(out)
        out = self.sigmoid(out)

        return out