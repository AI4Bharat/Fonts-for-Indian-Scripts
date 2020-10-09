import cv2
import random
import datetime
import numpy as np
import PIL.ImageOps
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


class ImageDataset(Dataset):
    
    def __init__(self, data_path, transform = None):
        self.train_data_path = data_path['data']
        self.input = os.listdir(os.getcwd() + '/hindi-data/content/Devanagari_Data_800/Input/Hindi/')
        self.non_italics_train_images = os.listdir(self.train_data_path + "Non_Italics/Hindi/")
        self.italics_train_images = os.listdir(self.train_data_path + "Italics/Hindi/")
        self.data = [self.non_italics_train_images, self.italics_train_images]
        self.font_type = ['Non_Italics/', 'Italics/']
        self.transform = transform
    
    def __len__(self):
        
        return self.non_italics_train_images.__len__() + self.italics_train_images.__len__()
    
    def __getitem__(self, indx):
        if indx >= len(self):
            raise Exception("Index should be less than {}".format(len(self)))
            
        random_indx = np.random.randint(0, 10) % 2
        label_name = self.data[random_indx][indx % self.data[random_indx].__len__()]
        if random_indx == 1:
            label_name = "%d_img_%s.png"%(int(label_name.split("_")[0]) % 10, label_name.split("_")[2].split(".")[0])
            
        font_indx = label_name.split("_")[0]
        inp_indx = label_name.split("_")[2].split(".")[0]
        
        label = Image.open(self.train_data_path + self.font_type[random_indx] + "Hindi/" + label_name).convert('L')
        inp = Image.open(os.getcwd() + '/hindi-data/content/Devanagari_Data_800/Input/Hindi/0_img_%s.png'%(inp_indx)).convert('L')
        
        rep_imgs = []
        
        for i in range(26):
            rep_indx = np.random.randint(0, 52)
            rep_imgs.append(Image.open(self.train_data_path + self.font_type[random_indx] + 'English/%s_img_%d.png'%(font_indx, rep_indx)).convert('L'))
        
        rep_imgs = np.stack(rep_imgs)
            
        
        if self.transform:
            inp = self.transform(inp)
            rep_imgs = self.transform(rep_imgs)
            label = self.transform(label)
            
        rep_imgs = rep_imgs.permute(2, 0, 1).permute(2, 0, 1)
            
        return inp, rep_imgs, label
