import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class BasicBlock(nn.Module):
    def __init__(self, channels = 128, stride = 1, padding = 1, norm_layer = 'BaN'):
        super(BasicBlock, self).__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.norm_layer = None
        
        self.conv_1 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels, kernel_size = 3, stride = self.stride, padding = self.padding)
        
        if norm_layer == 'BaN':
            self.norm_layer_1 = nn.BatchNorm2d(self.channels)
            self.norm_layer_2 = nn.BatchNorm2d(self.channels)
            
        elif norm_layer == 'InN':
            self.norm_layer_1 = nn.InstanceNorm2d(self.channels, affine = True)
            self.norm_layer_2 = nn.InstanceNorm2d(self.channels, affine = True)
            
        self.relu_1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels, kernel_size = 3, stride = self.stride, padding = self.padding)
        self.relu_2 = nn.PReLU()
        
    def forward(self, x):
        identity = x
        x = self.relu_1(self.norm_layer_1(self.conv_1(x)))
        x = self.norm_layer_2(self.conv_2(x)) + identity     
        return self.relu_2(x)

class CBnR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, padding = 0, stride = 1, activation = True, isCritic = False, norm_layer = 'BaN'):
        super(CBnR, self).__init__()
        
        self.activation = activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.isCritic = isCritic
        self.norm_layer = None
        
        self.conv_1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding)
        
        if norm_layer == 'BaN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm_layer == 'InN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine = True)
            
        self.relu = nn.PReLU()
        if self.isCritic:
            self.conv_last = nn.Conv2d(in_channels = out_channels, out_channels = out_channels//2, kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding)
        
    def forward(self, x):
        if self.activation:
            if self.isCritic:
                return self.conv_last(self.relu(self.norm_layer(self.conv_1(x))))
            else:
                return self.relu(self.norm_layer(self.conv_1(x)))
        else:
            return self.conv_1(x)

class TCBnR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, padding = 0, stride = 1, activation = True, norm_layer = 'BaN'):
        super(TCBnR, self).__init__()
        
        self.activation = activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm_layer = None
        
        self.conv_1 = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = self.kernel_size,
                                         stride = self.stride, padding = self.padding)
        if norm_layer == 'BaN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm_layer == 'InN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine = True)
        
        self.relu = nn.PReLU()
        
    def forward(self, x):
        if self.activation:
            return self.relu(self.norm_layer(self.conv_1(x)))
        else:
            return self.conv_1(x)
