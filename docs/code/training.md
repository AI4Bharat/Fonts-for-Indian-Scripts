Training Process for Model
==========================

We trained our model progressively consisting total 5 stages:

* **1st and 2nd Stage :** For 1st stage, we downsampled the input image from 320x320 to 20x20 and for 2nd stage the downsaampled size was 40x40.
* **3rd and 4th Stage :** From third stage, input dimentions of images was 80x80 alongwith this, we enabled the U-Net skip connections from Content Encoder to Generator. And same for the 4th stage with change in input image size to 160x160.
* **5th Stage :** In 5th stage we didn't downsampled the input image from it's original size of 320x320. In this stage we also disabled the U-Net connections.
* *For reference, you can refer these files <a href="https://github.com/AI4Bharat/Fonts-for-Indian-Scripts/blob/main/ProcessFont/Model/ConvNetworks.py">ConvNetworks.py</a>, <a href="https://github.com/AI4Bharat/Fonts-for-Indian-Scripts/blob/main/ProcessFont/Model/model.py">model.py</a> and <a href="https://github.com/AI4Bharat/Fonts-for-Indian-Scripts/blob/main/ProcessFont/Model/train.py"></a>*
