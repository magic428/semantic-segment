# Dataset

## PASCAL VOC 2012   

1. 访问 [下载链接](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal), 下载 **SBD** 和 **PASCAL VOC 2012** 数据集.  

2. 解压后得到两个文件夹: ***benchmark_RELEASE/*** 和 ***VOCdevkit/*** .    

3. 访问 [下载链接](
https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt), 下载 **seg11valid.txt** 文件, 将其保存在 ***VOCdevkit/VOC2012/ImageSets/Segmentation/*** 目录下.  

4. 将 ***benchmark_RELEASE/*** 和 ***VOCdevkit/*** 目录移动到 ***VOC/*** 目录下.  
5. 下载 VOC2012test.tar 数据集, 解压后得到 ***VOCdevkit/***, 将其重命名为  ***VOCdevkit-test/*** 后移动到 ***VOC/*** 目录下.  
5. 将 **voc.py** 脚本中的 **path** 变量设置为 ***VOC/*** 目录所在的根目录.  


## Cityscapes  

1. 从 cityscapes 官网下载以下数据: **leftImg8bit_trainvaltest**, **gtFine_trainvaltest**, **leftImg8bit_trainextra**, 和 **gtCoarse** .

2. 解压后将其全部放在 ***cityscapes/*** 文件夹下.     

3. 将 **cityscapes.py** 脚本中的 **path** 变量设置为 ***cityscapes/*** 目录所在的根目录.   
