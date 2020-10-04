import torch
import torch.nn as nn
import numpy as np
from model import StylisationNetwork
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Inference:

    def __init__(self, stage):

        self.transform_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485], [0.229]),])
        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229], std=[1/0.229])
        self.lmbda_l1 = 10
        self.lmbda_gp = 10

        self.styl_net = StylisationNetwork(self.lmbda_l1, self.lmbda_gp, norm_layer = 'InN')
        self.styl_net.load()
        self.styl_net.eval()
        self.stage = stage
        self.count = 0

    def stage_downsampling(self, batch, stage):
        input_destyle, style_images = batch
        input_destyle = nn.Upsample((2**stage)*10)(input_destyle)
        style_images = nn.Upsample((2**stage)*10)(style_images)
        return input_destyle, style_images

    def GenerateInputPairs(self, inpImg, fontImgs):
        fontImgs = np.stack(fontImgs)
        
        inpImg = self.transform_img(inpImg).unsqueeze(0)
        fontImgs = self.transform_img(fontImgs)

        fontImgs = fontImgs.permute(2, 0, 1).permute(2, 0, 1).unsqueeze(0)

        return inpImg, fontImgs

    def GenerateImage(self, inpImg, fontImgs):

        
        inpImg, fontImgs = self.stage_downsampling((inpImg, fontImgs), self.stage)

        inpImg = inpImg.to(device)
        fontImgs = fontImgs.to(device)

        x = self.styl_net.test(inpImg, fontImgs, self.stage)
        x = self.inv_normalize(x.detach()[0, :, :, :]).cpu().numpy().transpose(1, 2, 0).squeeze()
        
        return x
