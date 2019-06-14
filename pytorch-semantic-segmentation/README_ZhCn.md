# Semantic Segmentation 网络的 PyTorch 实现  

这个仓库中包含了一些 PyTorch 实现的 Semantic Segmentation 模型和训练代码.  

## 模型 - Models

1. Vanilla FCN: FCN32, FCN16, FCN8, in the versions of VGG, ResNet and DenseNet respectively
([Fully convolutional networks for semantic segmentation](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf))  

2. U-Net ([U-net: Convolutional networks for biomedical image segmentation](https://arxiv.org/pdf/1505.04597))  

3. SegNet ([Segnet: A deep convolutional encoder-decoder architecture for image segmentation](https://arxiv.org/pdf/1511.00561))  

4. PSPNet ([Pyramid scene parsing network](https://arxiv.org/pdf/1612.01105))  

5. GCN ([Large Kernel Matters](https://arxiv.org/pdf/1703.02719))  

6. DUC, HDC ([understanding convolution for semantic segmentation](https://arxiv.org/pdf/1702.08502.pdf))  

## 环境要求 - Requirement

1. PyTorch 0.2.0  

2. TensorBoard for PyTorch. [Here](https://github.com/lanpa/tensorboard-pytorch)  to install  

3. Some other libraries (find what you miss when running the code :-P)  

## 准备工作 - Preparation  

1. 进入 ***models/*** 目录, 设置 ***config.py*** 中的预训练模型路径;    

2. 进入 ***datasets/*** 目录,  根据 README 的提示操作.    

3. 下载 ***config.py*** 中提到的预训练模型.    

```
https://download.pytorch.org/models/vgg16-397923af.pth
https://download.pytorch.org/models/vgg19_bn-c79401a0.pth
https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth
https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
https://download.pytorch.org/models/resnet152-b121ed2d.pth
https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
https://download.pytorch.org/models/densenet201-c1103571.pth
```






## TODO

1. DeepLab v3 

2. RefineNet  

3. More dataset (e.g. ADE)  