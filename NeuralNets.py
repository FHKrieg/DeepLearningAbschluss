import torch
from torch import nn


# TODO Build models with other activation function 

#From class adapted
class NN_128_3(nn.Module):

    def __init__(self, input_size):
        super(NN_128_3, self).__init__()
                
        self.hidden_1 = nn.Linear(input_size, 128)
        self.hidden_2 = nn.Linear(128, 64)
        self.hidden_3 = nn.Linear(64, 32)
        self.output = nn.Linear(32,1)
        #from https://wandb.ai/authors/ayusht/reports/Implementing-Dropout-in-PyTorch-With-Example--VmlldzoxNTgwOTE
        self.dropout = nn.Dropout(0.25)
        
 
    def forward(self, x):

        out1 = nn.ReLU()(self.hidden_1(x))
        out2 = self.dropout(out1)
        out3 = nn.ReLU()(self.hidden_2(out2))
        out4 = self.dropout(out3)
        out5 = nn.ReLU()(self.hidden_3(out4))
        return self.output(out5)


class NN_256_4(nn.Module):

    def __init__(self, input_size):
        super(NN_256_4, self).__init__()
                
        self.hidden_1 = nn.Linear(input_size, 256)
        self.hidden_2 = nn.Linear(256, 128)
        self.hidden_3 = nn.Linear(128, 64)
        self.hidden_4 = nn.Linear(64, 32)
        self.output = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.25)
 
    def forward(self, x):

        out1 = nn.ReLU()(self.hidden_1(x))
        out2 = self.dropout(out1)
        out3 = nn.ReLU()(self.hidden_2(out2))
        out4 = self.dropout(out3)
        out5 = nn.ReLU()(self.hidden_3(out4))
        out6 = self.dropout(out5)
        out7 = nn.ReLU()(self.hidden_4(out6))

        return self.output(out7)


class NN_512_5(nn.Module):

    def __init__(self, input_size):
        super(NN_512_5, self).__init__()
                
        self.hidden_1 = nn.Linear(input_size, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.hidden_3 = nn.Linear(256, 128)
        self.hidden_4 = nn.Linear(128, 64)
        self.hidden_5 = nn.Linear(64, 32)
        self.output = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.25)
 
    def forward(self, x):

        out1 = nn.ReLU()(self.hidden_1(x))
        out2 = self.dropout(out1)
        out3 = nn.ReLU()(self.hidden_2(out2))
        out4 = self.dropout(out3)
        out5 = nn.ReLU()(self.hidden_3(out4))
        out6 = self.dropout(out5)
        out7 = nn.ReLU()(self.hidden_4(out6))
        out8 = self.dropout(out7)
        out9 = nn.ReLU()(self.hidden_5(out8))
        
        return self.output(out9)