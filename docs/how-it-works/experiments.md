**Comparisons**

[Final Results (With SSIM Scores)](https://drive.google.com/file/d/14u_GNi4MFkm7-X-Y6XmZWVPRl1ev5UYW/view?usp=sharing)


<table>
  <tr>
   <td><strong>Previously Used</strong>
   </td>
   <td><strong>Cons/Flaws</strong>
   </td>
   <td><strong>Alternative Chosen</strong>
   </td>
   <td><strong>Evidence</strong>
   </td>
  </tr>
  <tr>
   <td>Inner stages U-Net connections
   </td>
   <td>Leads to weak bottleneck training.
   </td>
   <td>Outer stages U-Net connections
   </td>
   <td><a href="https://drive.google.com/drive/folders/1FhL9TZWkPp6K-EJo6fTxs09rlig5akMQ?usp=sharing">Results</a>
   </td>
  </tr>
  <tr>
   <td>Batch Normalization
   </td>
   <td>Requires higher batch size for better results
   </td>
   <td>Instance Normalization
   </td>
   <td><a href="https://drive.google.com/drive/folders/1h-VePGyuIhoP1FIZ-xh_Pvch9oxKQv13?usp=sharing">Results</a>
   </td>
  </tr>
  <tr>
   <td>Concatenation of encoded embedding
   </td>
   <td>Becomes difficult for the decoder network to generalize. 
   </td>
   <td>Training a standalone Mixer Network.
   </td>
   <td> <a href="https://drive.google.com/drive/folders/16BSkqYb148MFZAS3lvDSvxzlatCREwn7?usp=sharing">Results</a>
   </td>
  </tr>
  <tr>
   <td>Single Image as font input
   </td>
   <td>Can’t capture font style
   </td>
   <td>Try out more images as input/
<p>
26/52 Randomly chosen images
   </td>
   <td> <a href="https://drive.google.com/drive/folders/1jNPwGqw2KThATokGYQaNikxTlipf0b3N?usp=sharing">Results</a>
   </td>
  </tr>
  <tr>
   <td>Few Images as font Input
   </td>
   <td>Can’t capture input font style
   </td>
   <td>26/52 Randomly chosen images
   </td>
   <td> <a href="https://drive.google.com/drive/folders/1PKrDptKHT-UdFkWyLo5zpsaRMeszFuFW?usp=sharing">Results</a>
   </td>
  </tr>
  <tr>
   <td>Without Progressive training
   </td>
   <td>Bottleneck layers aren’t trained properly
   </td>
   <td>With Progressive Training
   </td>
   <td> <a href="https://drive.google.com/drive/folders/1u9-XvfkTFiat2Sib1q_BkPZq5_wEcka8?usp=sharing">Results</a>
   </td>
  </tr>
  <tr>
   <td>Randomly picking any style as input to the content encoder
   </td>
   <td>Difficulties in generalization
   </td>
   <td>Fixing a particular style for input to the content encoder
   </td>
   <td> <a href="https://drive.google.com/drive/folders/1FhL9TZWkPp6K-EJo6fTxs09rlig5akMQ?usp=sharing">Results</a>
   </td>
  </tr>
  <tr>
   <td>Trained in 5 Indian Languages
   </td>
   <td>Difficulties in recreating input content
   </td>
   <td>Used only Devanagari data
   </td>
   <td><a href="https://drive.google.com/drive/folders/1esLDxkH0JnjBCJt2M-Ojd9ASgO8iouCw?usp=sharing">Results</a>
   </td>
  </tr>
  <tr>
   <td>Without Chinese Pretrained
   </td>
   <td>Difficulties in generalization on more styles
   </td>
   <td>Used a pretrained network on Chinese data.
   </td>
   <td><a href="https://drive.google.com/drive/folders/1ePigUruPb0np7WTUrCCLWNDGCdovenHR?usp=sharing">Results</a>
   </td>
  </tr>
</table>
