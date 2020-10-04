import os
import cv2
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


class FontUtil:

    def __init__(self, model):

        self.path = None
        self.word_list = None
        self.font = None
        self.inputFont = None
        self.unicodeCharSizes = None
        self.maxCharDimension = None
        self.padding = None
        self.gridSize = None
        self.fontImgs = None
        self.model = model.Inference(5)

    def ReadFont(self, rel_path, font_name):
        if self.path == None:
            self.path = rel_path
        if self.word_list == None:
            self.GetWordList()
        self.font = ImageFont.truetype(rel_path + font_name, 400, layout_engine=ImageFont.LAYOUT_RAQM)
        if self.inputFont == None:
            self.inputFont = ImageFont.truetype(rel_path + "/All TTFs/Input/Halant-Medium.ttf", 400, layout_engine=ImageFont.LAYOUT_RAQM)

    def InitParams(self, chr_list):
        self.unicodeCharSizes = map(lambda c: self.font.getsize(c), chr_list)
        self.maxCharDimension = max(map(lambda s: max(s), self.unicodeCharSizes))
        self.padding = 11
        self.gridSize = self.maxCharDimension + self.padding
        print(self.unicodeCharSizes, "\t", self.maxCharDimension, "\t" , self.padding, "\t", self.gridSize)

    def GetWordList(self):
        with open(self.path + '/Words List/words_list.pickle', 'rb') as file:
            self.word_list = pickle.load(file)
            print(len(self.word_list))

    def DrawEnglishChars(self, fig, ax, chr_list):
        count = 0
        engChars = []
        
        for i in range(4):
            for j in range(13):
                if count < 26:
                    
                    char = chr_list[0]
                    
                    if count == 0:
                        self.InitParams(char)
                    engImg = self.DrawString(char[count], 320, 320).squeeze()
                else:
                    char = chr_list[1]
                    
                    if count == 26:
                        self.InitParams(char)
                    engImg = self.DrawString(char[count-26], 320, 320).squeeze()
                engChars.append(engImg)
                ax[i][j].imshow(engImg, cmap='gray')
                ax[i][j].axis("off")
                count += 1

        count = 0
             
        fig.tight_layout()
        self.fontImgs = random.sample(engChars, 26)
        return fig, ax

    def GenerateWordImages(self, words):
        imgs = []
        genImgs = []
        for word in words:

            genImg = self.DrawString(word, 320, 320, inputFont = True)
            imgs.append(genImg)
            
            if self.fontImgs != None:
                inpImgs, fontImgs = self.model.GenerateInputPairs(genImg, self.fontImgs[0:26])
                outImg = self.model.GenerateImage(inpImgs, fontImgs)
                genImgs.append(outImg)
                
        img = np.concatenate(imgs, axis = 1)
        genImg = np.concatenate(genImgs, axis = 1)
        return img, genImg
        
        
    def DrawString(self, char, height, width, inputFont=False):
        font = None
        font = self.font
        if inputFont:
            font = self.inputFont
            self.InitParams(self.word_list)

        gridSize = self.gridSize
        x, y = font.getsize(char)
        theImage = Image.new('RGB', (gridSize, gridSize), color='white')
        theDrawPad = ImageDraw.Draw(theImage)
        theDrawPad.text(((gridSize-x)/2, (gridSize-y)/2), char, font=font, fill='black' )                  
        img = cv2.resize(np.asarray(theImage), (height, width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        return img
