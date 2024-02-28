import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import math

# Wirte a 1 D Conv Block with Batch Normalization and ReLU
# dropout_rate = 0.2
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super(ConvBlock1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
# create test data and output shape of output tensor
# conv = ConvBlock1D(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding="same")
# x = torch.randn(1, 3, 10)
# print(x)
# print(conv(x))
# print(conv(x).shape)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Write a 1D CNN with 3 Conv Blocks with Classification head
# kernel_size = 11
# out_channels = 64
pooling = 3
num_classes = 3
# max_length = 100

class CNN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                 num_classes, max_length, pe=False, dropout=0.2, pooling=1):
        super(CNN, self).__init__()
        self.pe = pe
        #add embedding layer
        self.embedding = nn.Embedding(in_channels, in_channels)
        print(in_channels)
        #add position encoding
        #self.position_encoding = PositionalEncoding(in_channels, dropout=dropout, max_len=max_length)
        # 3 Conv Blocks
        self.conv1 = ConvBlock1D(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = "same", dropout = dropout)
        self.conv2 = ConvBlock1D(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride= 1, padding = "same", dropout = dropout)
        self.conv3 = ConvBlock1D(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = "same", dropout = dropout)
        # global pooling
        self.pool = nn.AdaptiveAvgPool1d(pooling)
        self.classification_head = nn.Sequential(
            nn.Linear(out_channels*pooling, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes)
        )
        
    def forward(self, x):
        # x = self.embedding(x)
#         if self.pe:
#             x = x.permute(2, 0, 1)
#             x = self.position_encoding(x)
#             x = x.permute(1, 2, 0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.classification_head(x)
        return x

# write test data and output shape of output tensor
# cnn = CNN(in_channels=4, num_classes=num_classes)
# x = torch.randn(1, 4, 100)
# print(x.shape)
# # print(cnn(x))
# print(cnn(x).shape)




#############Autoencoder##############
class Encoder(nn.Module):
    def __init__(self, num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, shape):
        super(Encoder, self).__init__()
        self.shape = shape 
        
        assert self.shape[-1] / (2**3) % int(self.shape[-1] / (2**3)) == 0
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels*2, hidden_channels*2, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(int(self.shape[-1]*hidden_channels*2/stride**3), embedded_dims)  
        )
        
    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, shape):
        super(Decoder, self).__init__()
        self.shape = shape
        self.hidden_channels = hidden_channels
        
        self.shrunk_dims = int(self.shape[1]*hidden_channels*2/stride**3)
        self.linear =  nn.Sequential(
            nn.Linear(embedded_dims,self.shrunk_dims),  
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels*2, hidden_channels*2, kernel_size=kernel_size,\
                               stride=stride, padding=padding, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels*2, hidden_channels*2, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels*2, hidden_channels, kernel_size=kernel_size,\
                               stride=stride, padding=padding, output_padding=1),          
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_channels, self.shape[0], kernel_size=kernel_size,\
                               stride=stride, padding=padding, output_padding=1),
            nn.Sigmoid()  # Use sigmoid activation for the output to constrain values between 0 and 1
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0],  self.hidden_channels*2, int(self.shrunk_dims/(self.hidden_channels*2)))
        x = self.decoder(x)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, shape):
        super(Autoencoder, self).__init__()
        
        self.shape = shape 
        
        assert self.shape[-1] / (2**3) % int(self.shape[-1] / (2**3)) == 0
        self.encoder = Encoder(num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, shape)
        self.decoder = Decoder(num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, shape)

    def l1_regularization(self):
        l1_reg = 0.0
        for param in self.encoder.parameters():
            l1_reg += torch.sum(torch.abs(param))
        return l1_reg

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  # Adjust your_input_length
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    
class Encoder2D(nn.Module):
    def __init__(self, num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, shape):
        super(Encoder2D, self).__init__()
        self.shape = shape 
        assert self.shape[-1]/stride**3 % int(self.shape[-1]/stride**3) == 0
        assert self.shape[-2]/stride**3 % int(self.shape[-2]/stride**3) == 0
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Flatten(),
            nn.Linear(int(self.shape[-1]/stride**3)*int(self.shape[-2]/stride**3) * hidden_channels*2, embedded_dims)
        )
        
    def forward(self, x):
        return self.encoder(x)

    
class Decoder2D(nn.Module):
    def __init__(self, num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, output_padding, shape):
        super(Decoder2D, self).__init__()
        self.stride = stride
        self.shape = shape
        self.hidden_channels = hidden_channels
        
        assert self.shape[-1]/stride**3 % int(self.shape[-1]/stride**3) == 0
        assert self.shape[-2]/stride**3 % int(self.shape[-2]/stride**3) == 0

        self.shrunk_dims = int(self.shape[-1]/stride**3)*int(self.shape[-2]/stride**3) * hidden_channels*2
        self.linear =  nn.Sequential(
            nn.Linear(embedded_dims,self.shrunk_dims),  
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels*2, hidden_channels*2, kernel_size=kernel_size,\
                               stride=stride, padding=padding, output_padding=output_padding),
            nn.ReLU(),
            nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels*2, hidden_channels, kernel_size=kernel_size,\
                               stride=stride, padding=padding, output_padding=output_padding),          
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, self.shape[0], kernel_size=kernel_size,\
                               stride=stride, padding=padding, output_padding=output_padding),
            nn.Sigmoid()  # Use sigmoid activation for the output to constrain values between 0 and 1
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0],  self.hidden_channels*2, int(self.shape[-2]/self.stride**3), int(self.shape[-1]/self.stride**3))
        x = self.decoder(x)
        return x

class Autoencoder2D(nn.Module):
    def __init__(self, num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, output_padding, shape):
        super(Autoencoder2D, self).__init__()
        
        self.shape = shape 
        
        assert self.shape[-1] / (2**3) % int(self.shape[-1] / (2**3)) == 0
        self.encoder = Encoder2D(num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, shape)
        self.decoder = Decoder2D(num_channels, hidden_channels, kernel_size, stride, padding, embedded_dims, output_padding, shape)
        self.latent_norm = nn.LayerNorm(embedded_dims)

    def l1_regularization(self):
        l1_reg = 0.0
        for param in self.encoder.parameters():
            l1_reg += torch.sum(torch.abs(param))
        return l1_reg

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent_norm(x)
        x = self.decoder(x)  # Adjust your_input_length
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.latent_norm(x)
        return x