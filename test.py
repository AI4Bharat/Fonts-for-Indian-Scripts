import io
import os
import matplotlib.pyplot as plt
import ProcessFont.fonts as pf


def GetEnglishChars():
    chr_list = [chr(ord('A') + k) for k in range(26)]
    chr_list.extend([chr(ord('a') + k) for k in range(26)])
    return chr_list

fontUtil = pf.FontUtil()


fontUtil.ReadFont(os.getcwd(), "Biryani-Regular.ttf")
fig, ax = plt.subplots(4, 13, figsize=(40, 40))
fig, ax = fontUtil.DrawEnglishChars(fig, ax, GetEnglishChars())
plt.show()
    
