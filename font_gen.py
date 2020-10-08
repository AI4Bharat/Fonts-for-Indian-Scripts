import os
import json
import cv2
import numpy as np
import random
import shutil
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pickle

from PIL import features
print (features.check('raqm'))

with open('hi_word_list.json') as hindi:
    hindi_data = json.load(hindi)

consonants = [str(i) for i in range(904, 940)]
for i in range(len(consonants)):
    consonants[i] = r'\u0' + consonants[i] 

for i in 'ABCDEF':
    consonants.append(r'\u090' + i)
    consonants.append(r'\u091' + i)
    consonants.append(r'\u092' + i)

matras = ['\u0900', '\u0901', '\u0902', '\u0903', '\u093A', '\u093B', '\u093C', '\u093E','\u093F','\u0940','\u0941','\u0942','\u0943',
          '\u0944','\u0945','\u0946','\u0947','\u0948','\u0949','\u094A','\u094B','\u094C','\u094D',
          '\u094E','\u094F','\u0955','\u0956','\u0957','ॅ','॥',' ','﻿']

others = "1234567890qwertyuiopasdfghj\klzxcvbnm/.,QWERTYUIOPASDFGHJKLZXCVBNM!@#$%^&*()-_=+`~"

words_list = []
for word in hindi_data:

    words_list.append(word)
    





for word in words_list:
    for char in word:
        if char in others:
            try:
                words_list.remove(word)
            except:
                continue

word_list = []
count_list = [0]*14

for word in words_list:
    if word in ['प्डःडडडःडप्डडडडभ्ड्', 'गुबुगुबुगुबुगुबुगु','फिल्ल्निट्ॅ', 'आकांक्षाओंॅ']:
        continue
    if word.__len__() <= 14:
        if word[0] in ['ं', 'ः']:
            continue
        if word.__len__() == 1:
            if (word[0] in others) or (word[0] in matras):
                continue
        ln = word.__len__()
        if ln in [13, 14, 12, 11, 10]:
            if count_list[ln-1] <70:
                count_list[ln-1] += 1
                if not (word in word_list): 
                    word_list.append(word)

        else:
            if count_list[ln-1] <50:
                count_list[ln-1] += 1
                if not (word in word_list): 
                    word_list.append(word)
print(len(word_list))
with open('words_list.pickle', 'wb') as file:
    pickle.dump(word_list, file)
glyphs = {'Hindi' : [word_list], 'English' : [[chr(x) for x in range(65, 91)], [chr(x) for x in range(97, 123)]]}

root = 'New TTFs/'
out = 'Data/'
out1 = 'Data320/'

if os.path.exists(out) == False:
    os.mkdir(out)
    os.mkdir(out1)

for folder in os.listdir(root):

    if folder == 'Training':

        for style in os.listdir(root+folder):
            in_path = root+folder+'/'+style+'/'
            out_path = out+folder+'/'+style+'/'
            out_path1 = out1+folder+'/'+style+'/'

            if os.path.exists(out+folder) == False:
                os.mkdir(out+folder)
                os.mkdir(out1+folder)
                
            if os.path.exists(out_path) == False:
                os.mkdir(out_path)
                os.mkdir(out_path1)

            fonts = os.listdir(in_path)
            count = 0
            i=0
            j=0
            indx = 0
            print(fonts.__len__())


            for font_file in fonts:
                print(i)
                if i<112:
                    i+=1
                    continue
                else:
                    file_name = font_file.split('.')[0] + '/'
                    if font_file.split('.')[1] == 'ttf':
                        font = ImageFont.truetype(in_path + font_file, 400,layout_engine=ImageFont.LAYOUT_RAQM)

                        
                        for lang in glyphs:
                            if os.path.exists(out_path + lang) == False:
                                os.mkdir(out_path + lang)
                                os.mkdir(out_path1 + lang)

                            for glyph in glyphs[lang]:
                                

                                unicodeCharSizes = map(lambda c: font.getsize(c), glyph)
                                maxCharDimension = max(map(lambda s: max(s), unicodeCharSizes))
                                padding = 11
                                gridSize = maxCharDimension + padding
                                print(unicodeCharSizes, "\t", maxCharDimension, "\t" , padding, "\t", gridSize)
                                continue
                                for char in glyph:
                                    
                                    x, y = font.getsize(char)
                                    theImage = Image.new('RGB', (gridSize, gridSize), color='white')
                                    theDrawPad = ImageDraw.Draw(theImage)
                                    theDrawPad.text(((gridSize-x)/2, (gridSize-y)/2), char, font=font, fill='black' )
                                    ig = cv2.resize(np.asarray(theImage), (320, 320))
                                    ig = cv2.cvtColor(ig, cv2.COLOR_BGR2GRAY)
                                    plt.imsave(out_path1 + lang + '/' + str(i) +'_img_' + str(indx) + '.png', ig, cmap='gray')
                                    count += 1
                                    indx += 1
                                    if count%3000 == 0:
                                        print(count, ' Images done...')
                            indx = 0
                        i += 1
    else:
        in_path = root+folder+'/'
        out_path = out+folder+'/'
        out_path1 = out1+folder+'/'

        if os.path.exists(out_path) == False:
            os.mkdir(out_path)
            os.mkdir(out_path1)

        fonts = os.listdir(in_path)
        count = 0
        i=0
        j=0
        indx = 0
        print(fonts[:])


        for font_file in fonts:
            file_name = font_file.split('.')[0] + '/'
            if font_file.split('.')[1] == 'ttf':
                font = ImageFont.truetype(in_path + font_file, 400,layout_engine=ImageFont.LAYOUT_RAQM)

                    
                for lang in glyphs:
                    if os.path.exists(out_path + lang) == False:
                        os.mkdir(out_path + lang)
                        os.mkdir(out_path1 + lang)
                    for glyph in glyphs[lang]:

                        unicodeCharSizes = map(lambda c: font.getsize(c), glyph)
                        maxCharDimension = max(map(lambda s: max(s), unicodeCharSizes))
                        padding = 11
                        gridSize = maxCharDimension + padding
                        print(unicodeCharSizes, "\t", maxCharDimension, "\t" , padding, "\t", gridSize)
                        continue
                        for char in glyph:
                            
                            x, y = font.getsize(char)
                            theImage = Image.new('RGB', (gridSize, gridSize), color='white')
                            theDrawPad = ImageDraw.Draw(theImage)
                            theDrawPad.text(((gridSize-x)/2, (gridSize-y)/2), char, font=font, fill='black' )
                            ig = cv2.resize(np.asarray(theImage), (320, 320))
                            ig = cv2.cvtColor(ig, cv2.COLOR_BGR2GRAY)
                            plt.imsave(out_path1 + lang + '/' + str(i) +'_img_' + str(indx) + '.png', ig, cmap='gray')
                            count += 1
                            indx += 1
                            if count%3000 == 0:
                                print(count, ' Images done...')
                    indx = 0
                i += 1
