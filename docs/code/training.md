Training Process for Model
==========================

We trained our model progressively consisting total 5 stages:

* **1st and 2nd Stage :** For 1st stage, downsampled the input image from 320x320 to 20x20 and for 2nd stage the downsaampled size was 40x40
* **3rd and 4th Stage :** From third stage, the input image having dimentions 80x80 alongwith this, we enabled the U-Net skip connections from Content Encoder to Generator. And same for 4th stage with change in input image size to 160x160.
* **5th Stage :** In 5th stage we doesn't downsampled the input image with it's original size of 320x320. In this stage we also disabled the U-Net connections
