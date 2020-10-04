import torch
import torch.nn as nn
from basicBlock import CBnR, TCBnR
import torch.nn.functional as F
import torchvision.transforms as transforms


class Critic(nn.Module):
    def __init__(self, block, norm_layer = 'InN'):
        super(Critic, self).__init__()
        self.block = block
        self.x_conv_4 = None
        self.x_conv_5 = None
        
        self.input_conv = nn.Sequential(
                       CBnR(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, norm_layer = norm_layer)
                       )
        
        self.rgb2chnlBlock = nn.Sequential(
                       CBnR(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, norm_layer = norm_layer)
                       )
        
        self.layer_1 = self.make_layers(1, 128, norm_layer)
        self.layer_2 = self.make_layers(1, 256, norm_layer)
        self.layer_3 = self.make_layers(1, 256, norm_layer)
        self.layer_4 = self.make_layers(1, 256, norm_layer)
        self.layer_5 = self.make_layers(1, 256, norm_layer)
        
        self.downsample_conv_1 = CBnR(in_channels = 128, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.downsample_conv_2 = CBnR(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.downsample_conv_3 = CBnR(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.downsample_conv_4 = CBnR(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.downsample_conv_5 = CBnR(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2, isCritic = True, norm_layer = norm_layer)
        
    def make_layers(self, layers, channels, norm_layer):
        res_layers = []
        for i in range(layers):
            res_layers.append(self.block(channels = channels, norm_layer = norm_layer))
        return nn.Sequential(*res_layers)
    
    def stage_1(self, x, is_input = False):
        if is_input:
            x = self.rgb2chnlBlock(x)
        x = self.layer_5(x)
        x = self.downsample_conv_5(x) 
        return x
    
    def stage_2(self, x, is_input = False):
        if is_input:
            x = self.rgb2chnlBlock(x)
        x = self.layer_4(x)
        x = self.downsample_conv_4(x)
        return self.stage_1(x)
    
    def stage_3(self, x, is_input = False):
        if is_input:
            x = self.rgb2chnlBlock(x)
        x = self.layer_3(x)
        x = self.downsample_conv_3(x) 
        return self.stage_2(x)
    
    def stage_4(self, x, is_input = False):
        if is_input:
            x = self.rgb2chnlBlock(x)
        x = self.layer_2(x)
        x = self.downsample_conv_2(x) 
        return self.stage_3(x)
    
    def stage_5(self, x, is_input = False):
        x = self.input_conv(x)
        x = self.layer_1(x)
        x = self.downsample_conv_1(x) 
        return self.stage_4(x)
    
    def forward(self, x, stage):
        stage_funcs = [self.stage_1, self.stage_2, self.stage_3, self.stage_4, self.stage_5]
        return stage_funcs[stage - 1](x, is_input = True)\

######################################################################################################################

class Encoder(nn.Module):
    def __init__(self, block, inp_channel, norm_layer = 'InN'):
        super(Encoder, self).__init__()
        self.block = block
        self.conv_1 = None
        self.conv_2 = None
        self.conv_3 = None
        
        self.input_conv = nn.Sequential(
                       CBnR(in_channels = inp_channel, out_channels = 32, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, norm_layer = norm_layer)
                       )
        
        self.rgb2chnlBlock = nn.Sequential(
                       CBnR(in_channels = inp_channel, out_channels = 64, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, norm_layer = norm_layer)
                       )
        
        self.layer_1 = self.make_layers(1, 128, norm_layer)
        self.layer_2 = self.make_layers(1, 256, norm_layer)
        self.layer_3 = self.make_layers(1, 256, norm_layer)
        self.layer_4 = self.make_layers(1, 256, norm_layer)
        self.layer_5 = self.make_layers(1, 256, norm_layer)
        
        self.downsample_conv_1 = CBnR(in_channels = 128, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.downsample_conv_2 = CBnR(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.downsample_conv_3 = CBnR(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.downsample_conv_4 = CBnR(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.downsample_conv_5 = CBnR(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        
    def make_layers(self, layers, channels, norm_layer):
        res_layers = []
        for i in range(layers):
            res_layers.append(self.block(channels = channels, norm_layer = norm_layer))
        return nn.Sequential(*res_layers)
    
    def getConnections(self):
        return self.conv_2, self.conv_3
    
    def stage_1(self, x, is_input = False):
        if is_input:
            x = self.rgb2chnlBlock(x)
        x = self.layer_5(x)
        x = self.downsample_conv_5(x)
        return x
    
    def stage_2(self, x, is_input = False):
        if is_input:
            x = self.rgb2chnlBlock(x)
        x = self.layer_4(x)
        x = self.downsample_conv_4(x)
        return self.stage_1(x)
    
    def stage_3(self, x, is_input = False):
        if is_input:
            x = self.rgb2chnlBlock(x)
        x = self.layer_3(x)
        self.conv_3 = self.downsample_conv_3(x) 
        return self.stage_2(self.conv_3)
    
    def stage_4(self, x, is_input = False):
        if is_input:
            x = self.rgb2chnlBlock(x)
        x = self.layer_2(x)
        self.conv_2 = self.downsample_conv_2(x) 
        return self.stage_3(self.conv_2)
    
    def stage_5(self, x, is_input = False):
        x = self.input_conv(x)
        x = self.layer_1(x)
        x = self.downsample_conv_1(x) 
        return self.stage_4(x)
    
    def forward(self, x, stage):
        stage_funcs = [self.stage_1, self.stage_2, self.stage_3, self.stage_4, self.stage_5]
        return stage_funcs[stage - 1](x, is_input = True)

#################################################################################################################################

class Generator(nn.Module):
    def __init__(self, block, inp_enc, norm_layer = 'InN'):
        super(Generator, self).__init__()
        self.block = block
        self.inp_enc = inp_enc
        self.st_conv_1 = None
        self.st_conv_2 = None
        
        self.layer_1 = self.make_layers(1, 256, norm_layer)
        self.layer_2 = self.make_layers(1, 256, norm_layer)
        self.layer_3 = self.make_layers(1, 256, norm_layer)
        self.layer_4 = self.make_layers(1, 256, norm_layer)
        self.layer_5 = self.make_layers(1, 256, norm_layer)
        
        self.output_conv = nn.Sequential(
                       CBnR(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1, norm_layer = norm_layer),
                       CBnR(in_channels = 32, out_channels = 1, kernel_size = 3, padding = 1, activation = False, norm_layer = norm_layer)
                      )
        
        self.upsample_conv_1 = TCBnR(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.upsample_conv_2 = TCBnR(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.upsample_conv_3 = TCBnR(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.upsample_conv_4 = TCBnR(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        self.upsample_conv_5 = TCBnR(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2, norm_layer = norm_layer)
        
    
    def make_layers(self, layers, channels, norm_layer):
        res_layers = []
        for i in range(layers):
            res_layers.append(self.block(channels = channels, norm_layer = norm_layer))
        return nn.Sequential(*res_layers)
    
    def getConnections(self):
        self.st_conv_2, self.st_conv_3 = self.inp_enc.getConnections()
    
    def stage_1(self, x, is_input = False):
        x = self.upsample_conv_1(x) 
        x = self.layer_1(x)
        if is_input: 
            return self.output_conv(x)
        return x
    
    def stage_2(self, x, is_input = False):
        x = self.stage_1(x)
        x = self.upsample_conv_2(x)
        x = self.layer_2(x)
        if is_input: 
            return self.output_conv(x)
        return x
    
    def stage_3(self, x, is_input = False):
        x = self.stage_2(x)
        x = self.upsample_conv_3(torch.cat((x, self.st_conv_3), dim = 1))
        x = self.layer_3(x)
        if is_input: 
            return self.output_conv(x)
        return x
    
    def stage_4(self, x, is_input = False):
        x = self.stage_3(x)
        x = self.upsample_conv_4(torch.cat((x, self.st_conv_2), dim = 1))
        x = self.layer_4(x)
        if is_input: 
            return self.output_conv(x)
        return x
    
    def stage_5(self, x, is_input = False):
        x = self.stage_4(x)
        x = self.upsample_conv_5(x)
        x = self.layer_5(x)
        x = self.output_conv(x)
        return x
    
    def forward(self, x, stage):
        self.getConnections()
        stage_funcs = [self.stage_1, self.stage_2, self.stage_3, self.stage_4, self.stage_5]
        return stage_funcs[stage - 1](x, is_input = True)

#################################################################################################################################

class Mixer(nn.Module):
    def __init__(self, block, norm_layer = 'InN'):
        super(Mixer, self).__init__()
        self.block = block
        self.conv_1 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.layer_1 = self.make_layers(1, 256, norm_layer)
        self.layer_2 = self.make_layers(1, 256, norm_layer)
        
    def make_layers(self, layers, channels, norm_layer):
        res_layers = []
        for i in range(layers):
            res_layers.append(self.block(channels = channels, norm_layer = norm_layer))
        return nn.Sequential(*res_layers)
    
    def forward(self, inp1, inp2):
        x = self.conv_1(torch.cat([inp1, inp2], dim = 1))
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x    
