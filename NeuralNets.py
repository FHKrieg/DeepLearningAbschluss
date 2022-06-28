import torch
from torch import nn
#From class adapted
#Dropout layers excluded because overfitting for this dataset is not a big issue and they caused more problems in the regression than they improved the results
class NN_32_1(nn.Module):
    def __init__(self, input_size):
        super(NN_32_1, self).__init__()        
        self.hidden_1 = nn.Linear(input_size, 32)
        self.output = nn.Linear(32,1)
         
    def forward(self, x):
        out1 = nn.ReLU()(self.hidden_1(x))
        return self.output(out1)

class NN_64_2(nn.Module):

    def __init__(self, input_size):
        super(NN_64_2, self).__init__()        
        self.hidden_1 = nn.Linear(input_size, 64)
        self.hidden_2 = nn.Linear(64, 32)
        self.output = nn.Linear(32,1)
         
    def forward(self, x):

        out1 = nn.ReLU()(self.hidden_1(x))
        out2 = nn.ReLU()(self.hidden_2(out1))
        return self.output(out2)

class NN_128_3(nn.Module):

    def __init__(self, input_size):
        super(NN_128_3, self).__init__()
                
        self.hidden_1 = nn.Linear(input_size, 128)
        self.hidden_2 = nn.Linear(128, 64)
        self.hidden_3 = nn.Linear(64, 32)
        self.output = nn.Linear(32,1)
 
    def forward(self, x):
        out1 = nn.ReLU()(self.hidden_1(x))
        out2 = nn.ReLU()(self.hidden_2(out1))
        out3 = nn.ReLU()(self.hidden_3(out2))
        return self.output(out3)


class NN_256_4(nn.Module):

    def __init__(self, input_size):
        super(NN_256_4, self).__init__()
                
        self.hidden_1 = nn.Linear(input_size, 256)
        self.hidden_2 = nn.Linear(256, 128)
        self.hidden_3 = nn.Linear(128, 64)
        self.hidden_4 = nn.Linear(64, 32)
        self.output = nn.Linear(32,1)
    
 
    def forward(self, x):

        out1 = nn.ReLU()(self.hidden_1(x))
        out2 = nn.ReLU()(self.hidden_2(out1))
        out3 = nn.ReLU()(self.hidden_3(out2))
        out4 = nn.ReLU()(self.hidden_4(out3))

        return self.output(out4)


class NN_512_5(nn.Module):

    def __init__(self, input_size):
        super(NN_512_5, self).__init__()
                
        self.hidden_1 = nn.Linear(input_size, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.hidden_3 = nn.Linear(256, 128)
        self.hidden_4 = nn.Linear(128, 64)
        self.hidden_5 = nn.Linear(64, 32)
        self.output = nn.Linear(32,1)
 
    def forward(self, x):

        out1 = nn.ReLU()(self.hidden_1(x))
        out2 = nn.ReLU()(self.hidden_2(out1))
        out3 = nn.ReLU()(self.hidden_3(out2))
        out4 = nn.ReLU()(self.hidden_4(out3))
        out5 = nn.ReLU()(self.hidden_5(out4))
        
        return self.output(out5)

class NN_1024_6(nn.Module):

    def __init__(self, input_size):
        super(NN_1024_6, self).__init__()
                
        self.hidden_1 = nn.Linear(input_size, 1024)
        self.hidden_2 = nn.Linear(1024,512)
        self.hidden_3 = nn.Linear(512, 256)
        self.hidden_4 = nn.Linear(256, 128)
        self.hidden_5 = nn.Linear(128, 64)
        self.hidden_6 = nn.Linear(64, 32)
        self.output = nn.Linear(32,1)
 
    def forward(self, x):

        out1 = nn.ReLU()(self.hidden_1(x))
        out2 = nn.ReLU()(self.hidden_2(out1))
        out3 = nn.ReLU()(self.hidden_3(out2))
        out4 = nn.ReLU()(self.hidden_4(out3))
        out5 = nn.ReLU()(self.hidden_5(out4))
        
        return self.output(out5)