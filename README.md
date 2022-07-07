# AR2OR-PAM
This repository stores codes for our published paper "[High-resolution photoacoustic microscopy with deep penetration through learning](https://www.sciencedirect.com/science/article/pii/S2213597921000732)"


## Abstract
Optical-resolution photoacoustic microscopy (OR-PAM) enjoys superior spatial resolution and has received intense attention in recent years. The application, however, has been limited to shallow depths because of strong scattering of light in biological tissues. In this work, we propose to achieve deep-penetrating OR-PAM performance by using deep learning enabled image transformation on blurry living mouse vascular images that were acquired with an acoustic-resolution photoacoustic microscopy (AR-PAM) setup. A generative adversarial network (GAN) was trained in this study and improved the imaging lateral resolution of AR-PAM from 54.0 µm to 5.1 µm, comparable to that of a typical OR-PAM (4.7 µm). The feasibility of the network was evaluated with living mouse ear data, producing superior microvasculature images that outperforms blind deconvolution. The generalization of the network was validated with in vivo mouse brain data. Moreover, it was shown experimentally that the deep-learning method can retain high resolution at tissue depths beyond one optical transport mean free path. Whilst it can be further improved, the proposed method provides new horizons to expand the scope of OR-PAM towards deep-tissue imaging and wide applications in biomedicine.  

## Integrated OR- and AR-PAM system
![](https://github.com/Ford666/AR2OR-PAM/blob/main/images/1.png)

## WGAN-GP model used for PAM imaging transformation
![](https://github.com/Ford666/AR2OR-PAM/blob/main/images/3.png)

## Deblurring performance on *in vivo* mouse ear vascular image
![](https://github.com/Ford666/AR2OR-PAM/blob/main/images/5.png)

## Application on mouse brain imaging
![](https://github.com/Ford666/AR2OR-PAM/blob/main/images/7.png)


If the codes are helpful for your research, please consider to cite out work:
> @article{cheng2022high,
  title={High-resolution photoacoustic microscopy with deep penetration through learning},
  author={Cheng, Shengfu and Zhou, Yingying and Chen, Jiangbo and Li, Huanhao and Wang, Lidai and Lai, Puxiang},
  journal={Photoacoustics},
  volume={25},
  pages={100314},
  year={2022},
  publisher={Elsevier}
}
