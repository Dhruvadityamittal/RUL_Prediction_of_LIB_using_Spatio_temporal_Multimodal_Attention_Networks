import math
import torch.nn as nn
import torch.nn.functional as F
from TinyHAR import TinyHAR_Model
import torch
import math

# CNN Model for Stage 1
class CNN_Model(nn.Module):
    
    def __init__(self,input_size, channels):
        super(CNN_Model, self).__init__()
        self.name = "CNN_Classifier"
        filter_size_1 = 21
        filter_size=21
        
        self.conv1 = nn.Conv1d(channels,16,kernel_size = filter_size_1, stride=1,padding=filter_size_1//2)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(2)
        

        self.conv2 = nn.Conv1d(16,32, kernel_size = filter_size_1, stride = 1,padding=filter_size_1 //2)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32,64, kernel_size = filter_size, stride = 1,padding=filter_size //2)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool1d(2)

       
        self.flatten_size = 128*math.floor(input_size/(2*2*2*2))
        self.flatten = nn.Flatten(start_dim=1)
        
        self.Linear1 = nn.Linear(self.flatten_size, input_size)
        self.batch_norm_linear = nn.BatchNorm1d(input_size)
        # self.a = nn.Linear()
        self.Linear2 = nn.Linear(input_size,1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)
        # print(self.flatten_size)
        
    
    def forward(self,x):
        # x= x.view(x.shape[0],1,x.shape[1])
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.batch_norm1(out)
        out = self.dropout(out)
        out = self.max_pool1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.batch_norm2(out)
        out = self.dropout(out)
        out = self.max_pool2(out)   

        out = self.conv3(out)
        out = self.relu(out)
        out = self.batch_norm3(out)
        out = self.dropout(out)
        out = self.max_pool3(out) 

        out = self.flatten(out)
        
        out = self.Linear1(out)  
        out = self.Linear2(out)

        out = self.sigmoid(out)
        return out


# LSTM Model for Stage 1
class LSTM_Model(nn.Module):
    
    def __init__(self,input_size,channels):
        super(LSTM_Model, self).__init__()
        self.name = "LSTM_Classifier"
        hidden_size1 = input_size
        hidden_size2 = input_size
        num_layers = 4
        self.LSTM1 = nn.LSTM(input_size = input_size, hidden_size = hidden_size1, num_layers = num_layers,batch_first=True)
        self.LSTM2 = nn.LSTM(input_size = input_size, hidden_size = hidden_size2, num_layers = num_layers,batch_first=True)

        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(hidden_size2*channels,128)
        self.Linear2 = nn.Linear(128,1)
        self.Linear3 = nn.Linear(50,1)
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
#         out = self.relu(out)
#         out = self.Linear3(out)
        out = self.sigmoid(out)
        return out


# CNN Model for Stage 2
class CNN_Model_RUL(nn.Module):
    
    def __init__(self,input_size, channels):
        super(CNN_Model_RUL, self).__init__()
        self.name = "CNN"
        filter_size_1 = 21
        filter_size   = 21
        
        self.conv1 = nn.Conv1d(channels,16,kernel_size = filter_size_1, stride=1,padding=filter_size_1//2)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(2)
        

        self.conv2 = nn.Conv1d(16,32, kernel_size = filter_size_1, stride = 1,padding=filter_size_1 //2)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32,64, kernel_size = filter_size, stride = 1,padding=filter_size //2)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool1d(2)

       
        self.flatten_size = 128*math.floor(input_size/(2*2*2*2))
        self.flatten = nn.Flatten(start_dim=1)
        
        self.Linear1 = nn.Linear(self.flatten_size, input_size)
        self.batch_norm_linear = nn.BatchNorm1d(input_size)
        # self.a = nn.Linear()
        self.Linear2 = nn.Linear(input_size,1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        # print(self.flatten_size)
      
    def forward(self,x):
        # x= x.view(x.shape[0],1,x.shape[1])
        
        out = self.conv1(x)
        out = self.relu(out)
        # out = self.batch_norm1(out)
        out = self.dropout(out)
        out = self.max_pool1(out)

        out = self.conv2(out)
        out = self.relu(out)
        # out = self.batch_norm2(out)
        out = self.dropout(out)
        out = self.max_pool2(out)   

        # out = self.conv3(out)
        # out = self.relu(out)
        # out = self.batch_norm3(out)
        # out = self.dropout(out)
        # out = self.max_pool3(out) 

        out = self.flatten(out)
        
        out = self.Linear1(out)  
        out = self.Linear2(out)
#         out = self.sigmoid(out)

        return out

# LSTM model for Stage 2
class LSTM_Model_RUL(nn.Module):
    
    def __init__(self,input_size,channels):
        super(LSTM_Model_RUL, self).__init__()
        self.name = "LSTM"
        hidden_size1 = input_size
        hidden_size2 = input_size
        num_layers = 4
        self.LSTM1 = nn.LSTM(input_size = input_size, hidden_size = hidden_size1, num_layers = num_layers,batch_first=True)
        # self.LSTM2 = nn.LSTM(input_size = input_size, hidden_size = hidden_size2, num_layers = num_layers,batch_first=True)

        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(hidden_size2*channels,128)
        self.Linear2 = nn.Linear(128,1)
        self.Linear3 = nn.Linear(50,1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        
        # self.h0 = torch.randn(4, x.size(0), 100)
        # self.c0 = torch.randn(4, x.size(0), 100)
        
        # x= x.view(x.shape[0],channels,x.shape[1])
        # out, (hn, cn) = self.LSTM(x, (self.h0, self.c0))

        out, (_, _) = self.LSTM1(x)
        # out,(_,_) = self.LSTM2(out)
        
        out = self.flatten(out)
        out = self.relu(out)
        out = self.Linear1(out)
        out = self.relu(out)
        out = self.Linear2(out)
#         out = self.relu(out)
#         out = self.Linear3(out)
#         out = self.sigmoid(out)


        return out
    

# Tranfromer Encode model for Stage 2    
class Autoencoder(nn.Module):
    def __init__(self,channels, input_size=16, hidden_dim=8, noise_level=0.01, kernel_size=21):
        super(Autoencoder, self).__init__()
        self.channels, self.input_size, self.hidden_dim, self.noise_level = channels, input_size, hidden_dim, noise_level
        
        # self.enc_conv1 = nn.Conv1d(channels, 5, kernel_size = kernel_size, padding = kernel_size//2)
        # self.enc_conv2 = nn.Conv1d(5, 5, kernel_size = kernel_size, padding = kernel_size//2)
        
        self.enc_fc = nn.Linear(input_size* channels, hidden_dim)
        
        self.dec_fc = nn.Linear(hidden_dim, input_size*channels)
        self.dec_conv1 = nn.Conv1d(5, 5, kernel_size = kernel_size, padding = kernel_size//2)
        self.dec_conv2 = nn.Conv1d(5, channels, kernel_size = kernel_size, padding = kernel_size//2)
        self.dec_conv_out = nn.Conv1d(channels, channels, kernel_size = 1)
        # self.dec_conv = nn.Conv1d(1, channels, kernel_size = kernel_size, padding = kernel_size//2)

        self.conv1 = nn.Conv1d(channels,16,kernel_size = 21, stride=1,padding=21//2)
        self.max_pool = nn.MaxPool1d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv2 = nn.Conv1d(16,32,kernel_size = 21, stride=1,padding=21//2)
        
#         flatten_size = 128*math.floor(50/(2*2*2))
        self.fc1 = nn.Linear(16*50, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_size)
        self.flatten = nn.Flatten(start_dim=1)
        self.relu  = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.en_conv1 = nn.Conv1d(32,16,kernel_size = 21, stride=1,padding=21//2)
        self.en_conv2 = nn.Conv1d(16,channels,kernel_size = 21, stride=1,padding=21//2)
    
        
    def encoder(self, x):
        # x = self.enc_conv1(x)  
        # x = self.relu(x)
        # x = self.max_pool(x)
        x = self.flatten(x)
        x = self.enc_fc(x)   # (8)
        

        # x = self.conv1(x)
#         x = self.relu(x/)
#       x = self.max_pool(x)
#         x = self.conv2(x)
#       x = self.max_pool(x)
        
               
        # x = self.flatten(x)
#         print(x.shape)
        h1 =self.relu(x)
        return h1
    
    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
        return corrupted_x
    
    def decoder(self, x):
        x = self.dec_fc(x)  # 50
        x = x.reshape(-1, self.channels, self.input_size)
        # x = self.up(x)
        # x = self.dec_conv2(x)
        # x = self.leaky_relu(x)
        # x = self.dec_conv_out(x)

        # x = self.relu(x)
        # x = torch.unsqueeze(x, 1)  # (1,50)
        
       
#         print(x.shape)
#         x = self.en_conv1(x)
#         x = self.relu(x)
        # x = self.en_conv2(x)
        # x = self.relu(x)

        return x
    
    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        
        decode = self.decoder(encode)
        # print('encode:', encode.size(), 'decode:', decode.size())
        # exit()
        return encode, decode
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=16):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class Net(nn.Module):
    def __init__(self, channels, feature_size=16, hidden_dim=128, num_layers=2, nhead=32, dropout=0.1, noise_level=0.01):
        super(Net, self).__init__()
        self.name = "Transformer"
        self.auto_hidden = int(feature_size/2)
        input_size = self.auto_hidden 
        self.pos = PositionalEncoding(d_model=input_size, max_len=input_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear1 = nn.Linear(input_size, 128)
        
        # self.linear2 = nn.Linear(128, 50)
        self.linear2 = nn.Linear(128, 1)
        self.linear3 = nn.Linear(50, 1)
#         self.LayerNorm = nn.LayerNorm(1)
        
        self.autoencoder = Autoencoder(channels,input_size=feature_size, hidden_dim=self.auto_hidden, noise_level=noise_level)
        self.sigmoid = nn.Sigmoid()
        self.relu    = nn.ReLU()
        self.fc1 = nn.Linear(feature_size*50, int(feature_size/2))
        
 
    def forward(self, x): 
        
        batch_size, feature_num, feature_size  = x.shape 
#         encode, decode = self.autoencoder(x.reshape(batch_size, -1))# batch_size*seq_len
        
        encode, decode = self.autoencoder(x)# batch_size*seq_len
        
        # print('encode:',encode.size())
        out = encode.reshape(batch_size, -1, self.auto_hidden)
        # print('reshape:',out.size())
        
        out = self.pos(out)
        # print("pos",out.shape)
        out = out.reshape(1, batch_size, -1) # (1, batch_size, feature_size)
        # print("pos reshape",out.shape)
        out = self.cell(out)  
        # print("cell",out.shape)
        
        out = out.reshape(batch_size, -1) # (batch_size, hidden_dim)

        out = self.relu(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)

        # out = self.linear1(out)            # out shape: (batch_size, 1)
        # out = self.relu(out)
#         print("fc3",out.shape)
#         out = self.LayerNorm(out)
        # out = self.linear2(out)
#         print("fc4",out.shape)
        # out = self.relu(out)
        
        # out = self.linear3(out)
#         print("fc5",out.shape)
#         out= self.sigmoid(out)
        
        return out     , decode

class Net_new(nn.Module):
    def __init__(self):
        super(Net_new, self).__init__()
        self.pos = PositionalEncoding(d_model=50, max_len=50)
        encoder_layers = nn.TransformerEncoderLayer(d_model=50, nhead=2, dim_feedforward=100, dropout=0.1)
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.conv1 = nn.Conv1d(7,16,kernel_size = 21, stride=1,padding=21//2)
        self.max_pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16,32,kernel_size = 21, stride=1,padding=21//2)
        self.flatten = nn.Flatten()
        flatten_size = 128*math.floor(50/(2*2*2*2))
        self.linear1 = nn.Linear(flatten_size,100)
        self.linear2 = nn.Linear(100,1)
        self.sigmoid = nn.Sigmoid()
    
    
    
    def forward(self,x):
        out = self.pos(x)
        out = self.cell(out)
        out = self.conv1(out)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = self.max_pool1(out)
        out = self.flatten(out)
        out = self.linear1(out)           
#         out = self.linear2(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        
        return out



# Tiny HAR model for Stage 2
class TransformerLSTM(nn.Module):
    def __init__(self, in_size, win_size):
        super(TransformerLSTM, self).__init__()
        # B F L C
        self.name = "TransformerLSTM"
        self.model = TinyHAR_Model(
            input_shape = (64, 1, win_size, in_size),
            number_class = 1,
            filter_num = 20,
            cross_channel_interaction_type = "attn",
            cross_channel_aggregation_type = "FC",
            temporal_info_interaction_type = "lstm",
            temporal_info_aggregation_type = "tnaive"
        )


    def forward(self, x):

        # B, C, L -> B, 1, C, L
        x = torch.unsqueeze(x, 1)

        # B, 1, C, L -> B, 1, L, C
        x = torch.permute(x, (0, 1, 3, 2) )

        # B, 1, L, C -> B, n_classes
        x = self.model(x)

        # BCE loss
        if len(x.shape) == 1:
            return x.squeeze()
        return x

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r = self.forward(x)
        # BCE
        if len(r.squeeze().shape)==1:
            return (torch.sigmoid(r)>0.5).long()
        return r.argmax(dim=1, keepdim=False)