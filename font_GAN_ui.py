import sys
import os
sys.path.append(os.getcwd() + "/ProcessFont/Model")
import ProcessFont.Model.inference as model
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
import ProcessFont.fonts as pf
import streamlit as stl

stl.set_option('deprecation.showfileUploaderEncoding', False)
fontUtil = pf.FontUtil(model)

def setLayout():
    stl.markdown(        f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 70%;
        }}
    </style>
    """
    , unsafe_allow_html=True,
    )

def dump_file(file_dump: io.BytesIO, format, output_folder):
    file = "/Uploaded Fonts/" + "Font" + format
    out_file = os.getcwd() + file
    with open(out_file, 'wb') as f:
        f.write(file_dump.getbuffer())
    return file

def GetAvailableFontList():
    return os.listdir(os.getcwd() + '/All TTFs/Avail_Fonts/')

def GetEnglishChars():
    chr_list = []
    chr_list.append([chr(ord('A') + k) for k in range(26)])
    chr_list.append([chr(ord('a') + k) for k in range(26)])
    return chr_list

def PrintCharacters(font_name):
    fontUtil.ReadFont(os.getcwd(), font_name)
    fig, ax = plt.subplots(4, 13, figsize=(40, 20))
    fig, ax = fontUtil.DrawEnglishChars(fig, ax, GetEnglishChars())
    stl.pyplot()
    

setLayout()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
stl.title("Font GAN for Indian Scripts")


def getInput():
    choice = stl.radio("Select your Options", ("Upload your own font file", "Select from Our list"))


    if choice == 'Upload your own font file':
        ttf_file = stl.file_uploader("Upload Your Font File")
        if ttf_file:
            font_path = dump_file(ttf_file, '.ttf', 'uploaded_fonts/')
            PrintCharacters(font_path)
            
    elif choice == "Select from Our list":
        font_name = stl.selectbox("Select Your Font", GetAvailableFontList())
        PrintCharacters("/All TTFs/Avail_Fonts/" + font_name)
    
getInput()
inp = stl.text_input("Type Hindi Text Here")
words = inp.split(" ")
inp_len = words.__len__()
if inp != "":
    if inp_len == 1:
        img, genImg = fontUtil.GenerateWordImages(words)
        genImg = np.clip(genImg, a_min = 0.0, a_max = 1.0)
        stl.image(img, caption="Input Image", width=200)
        stl.image(genImg, caption="Generated Image", width=200)
    else:
        img, genImg  = fontUtil.GenerateWordImages(words)
        genImg = np.clip(genImg, a_min = 0.0, a_max = 1.0)
        stl.image(img, caption="Input Image", width=400)
        stl.image(genImg, caption="Generated Image", width=400)
        
stl.text(inp)
    
