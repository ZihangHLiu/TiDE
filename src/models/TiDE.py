'''
This file contains the implementation of the TiDE model.
Original paper: https://arxiv.org/pdf/2304.08424.pdf
'''

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    '''
    The Residual block is the basic building block of TiDE.
    In feature projection, the out_features, is usually lower than in_features.
    '''
    def __init__(self, in_features, hid_features, out_features, dropout_prob=0.5):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hid_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hid_features, out_features)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.LayerNorm(out_features)

        # Shortcut connection
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out += self.shortcut(residual)
        out = self.norm(out)
        return out
    

class TiDEEncoder:
    '''
    The encoder block of TiDE.
    Consists of a stack of residual blocks.

    Note:
    The encoder internal layer sizes are all set to hiddenSize and 
    the total number of layers in the encoder is set to ne(numEncoderLayers)

    The input to the encoder layer has to be stacked and flattened, 
    past and future projected covariates: X(1, L + H) has to be projected to lower dimension
    '''
    def __init__(self, in_features, hid_features, out_features, num_blocks=1):
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(in_features, hid_features, out_features))
            in_features = out_features

    def forward(self, x):
        '''
        Here x is the concatenation of the following three elements:
        1. past and future projected covariates: X(1, L + H)
        2. past of the time seires: Y(1, L)
        3. static covariates: a(i)
        '''
        for block in self.blocks:
            x = block(x)

        return x
    

class TiDEDenseDecoder:
    '''
    The dense decoder block of TiDE.
    It maps the encoded hidden representations into future predictions of time series
    Also consists of a stack of residual blocks.

    Note:
    The hidden layer size of the dense encoder is the same as the hidden layer size of the encoder.
    The decoder output size has to be a vector with size p * H, where H is future time steps and p is the expected dimension of decoder output of a single time step vector.
    '''
    def __init__(self, in_features, hid_features, out_features, num_blocks=1):
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(in_features, hid_features, out_features))
            in_features = out_features

    def forward(self, x):
        '''
        Here x is the encoded hidden representation of the time series.
        '''
        for block in self.blocks:
            x = block(x)

        return x
    
class TiDETemporalDecoder:
    '''
    The temporal decoder block of TiDE.

    A residual block with output size 1 (the prediction), 
    that maps the decoded vector at time step t concatenated with the corresponding projected covariate.

    This can be useful if some covariated have a strong direct effect on a particular time-step's actual value
    '''
    def __init__(self, in_features, hid_features):
        self.residual = ResidualBlock(in_features, hid_features, 1)

    def forward(self, x):
        '''
        Here x is the concatenation of the following two elements:
        1. decoded vector at time step t: y(t)
        2. corresponding projected covariate: X(t)
        '''
        return self.residual(x)