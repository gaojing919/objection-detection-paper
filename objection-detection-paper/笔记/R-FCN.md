R-FCN: Object Detection via Region-based Fully Convolutional Networks
# 摘要 #
Fast/Faster R-CNN需要使用代码昂贵的per-region子网络几百次，R-FCN是全卷积的，所有的计算在整个图片上都是共享的  
使用位敏得分图去解决分类中和检测中的平移不变性的问题  
R-FCN可以很自然的使用分类的主干网络用于检测，比如文中使用的是ResNets  
代码位置：[https://github.com/daijifeng001/r-fcn](https://github.com/daijifeng001/r-fcn)   
# 1.介绍#
流行的目标检测网络由RoI吃化层分为两部分：跟RoI无关的共享的全卷积网络，跟RoI有关的不共享计算的子网络   
目前先进的分类网络ResNets，ResNets都是设计成全卷积的，使跟RoI相关的子网络没有隐藏层   
直接这样用于检测网络奖却屡很差，不能跟分类的精确率匹配  
ResNet解决这个问题的方法是：在两组卷积层之间插入faster rcnn的RoI池化层，提高精度，但由于没有更像RoI的早期计算，所以速度比较慢  
共享的全卷积架构跟FCN是一样的  
为了给FCN中加入平移不变性，，我们通过使用一些特定的卷积层构建一些位敏得分图作为FCN的输出  
每个位敏得分图都编码了相对（物体左上角）的空间位置信息   
在FCN的顶端加上了位敏RoI池化层照看得分图的信息   
# 2.我们的方法 #
跟faster rcnn一样，采用两个阶段的目标检测，候选域产生和域分类   
基于候选域的方法精确度更高  
跟faster rcnn一样，使用RPN产生候选域，RPN跟R-FCN共享卷积特征  
在R-FCN中，所有可学习权重的层都是卷积的，并且计算是基于整个图片的   
最后一个卷积层为每个类产生\\(k^2\\)个位敏得分图，最后一共输出\\(k^2(C+1)\\)  
\\(k^2\\) 个位敏得分图相当于一个\\(k\times k\\)的描述相对位置的空间网格   
R-FCN是以位敏RoI池化层结束的，这个层聚集了最后一个卷积层的输出，没每个RoI产生分数  
位敏RoI池化层是有选择性的池化，\\(k\times k\\)的每个bin只收集来自一个得分图的响应   
## 主干网络 ##
R-FCN是基于ResNet-101：100个卷积层，后面一个平均池化层和一个1000类的fc层，RFCN中移除了平均池化层和fc层  
ResNet-101最后一个卷积层的输出是2048维的，作者给后面加了一个随机初始化的1024维的卷积层，用来减少维度，然后使用\\(k^2(C+1)\\) 通道的卷积层产生得分图   
## 位敏得分图和位敏RoI池化 ##
 把每个RoI矩形用网格分割成\\(k\times k\\)个Bins
  ![](https://i.imgur.com/a9zxPBI.png)    
![](https://i.imgur.com/Jacliab.png)  投票   
![](https://i.imgur.com/MDyfyMc.png)  softmax响应   
增加一个\\(4k^2\\)维的姊妹卷积层用于Bounding-box回归 ，位敏RoI池化层作用在 \\(4k^2\\)的map上，产生一个\\(4k^2\\)的向量，然后通过平均投票得到一个4维向量，\\(t=(t_x,t_y,t_w,t_h)\\),这是类无关的bounding-box回归   
## 训练 ##
![](https://i.imgur.com/J1b3KVB.png)  
![](https://i.imgur.com/6DG9pUy.png)  
回归损失跟faster rcnn中的定义一样，\\(\lambda=1\\),正例是跟ground-truth的IoU至少是0.5的，否则定义为负类  
假设每张图片有N个候选框，前向传播时，计算N各候选框的损失，然后把所有的RoI按照损失排序，选出前B个最高的损失，后向传播只基于这B个损失   
训练是使用单尺度的，图片的大小调整为600像素    
跟faster rcnn一样采用4步训练，在训练RPN和R-FCN之间迭代的训练
## 推理 ##
RPN处理RoIs,R-FCN评估类得分和bounding box 回归  
## À trous and stride ##
由ResNet101的32 p的stride变为16 pixels，增加了score map的分辨率，前四个阶段的stride不变，第五阶段由stride=2变为1，其filter使用hole algorithm修改，其map可提高2.6个百分点：   
## 可视化 ##
# 3.相关工作 #
# 4.实验 #
## PASCAL VOC ##
下面的结果都是在训练集是07+12，测试集是07  
Naïve Faster R-CNN:卷积层使用ResNet-101去计算共享的feature map,最后一个卷积层后使用RoI池化，21-class fc层用于评估每个RoI,使用à trous trick  68.9%，比在Faster RCNN中使用ResNet（76.4%）要差  
Class-specific RPN：用ResNet-101 conv5训练RPN,使用à trous trick    
R-FCN without position-sensitivity：通过设置k=1,相当于对每个RoI做了个全局池化  ，R-FCN不能收敛  
## 根ResNet-101结构的faster rcnn比较 ##
\\(k\times k=7\times 7\\)   
faster rcnn为每个候选域用了个10层的网络去实现好的精度  
![](https://i.imgur.com/yk9lvQb.png)     
          