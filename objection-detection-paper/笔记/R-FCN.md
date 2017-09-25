R-FCN: Object Detection via Region-based Fully Convolutional Networks
# 摘要 #
Fast/Faster R-CNN需要使用代码昂贵的per-region子网络几百次，R-FCN是全卷积的，所有的计算在整个图片上都是共享的  
使用位敏得分图去解决分类中和检测中的平移不变性的问题  
R-FCN可以很自然的使用分类的主干网络用于检测，比如文中使用的是ResNets  
代码位置：[https://github.com/daijifeng001/r-fcn](https://github.com/daijifeng001/r-fcn)   
# 1.介绍 #