Model Architecture
==================

Our architecture consists of different types of Encoder and Decoder Convolutional Networks as shown below:
![Image](/docs/Images/arch.png)

Our Architecture
--------------------
1. Our architecture consists of 4 different parts :
2. Style Encoder and Content Encoder Network
3. Mixer Network
4. Generator Network
5. Critic Network

* **Style Encoder and Content Encoder Network :** It consists of two different types of encoder although the overall architecture of these two encoders are the same. An Encoder is a Convolutional Neural Network used for downsampling the input images. This downsampling is achieved using several ResNet layers while learning and mapping  the representation of input vector from n-dimensional vector space to m-dimensional vector space. The two encoder are style encoder and content encoder, Style encoder takes the 26 input images of 26,320,320 whereas Content Encoder takes a single input image of hindi glyph having 320,320 image dimensions to which that stylisation has to be done. The output dimensions of both the encoders are the same i,e; 1,10,10.

* **Mixer Network :** It has two inputs, one from style encoder and other from content encoder and applies a bilinear function on both the representations which generates a single output vector of 4,1,10,10.

* **Generator network :** Generator is also a Convolutional Neural Network which upsamples the input which inturn generates the output image of 320x320. In this network, instead of using convolution operation we have used transposed convolution operation which upsamples the input image.
