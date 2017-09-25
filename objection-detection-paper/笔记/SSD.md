SSD: Single Shot MultiBox Detector  
# 摘要 #
SSD:没有候选框产生阶段  
# 1.介绍 #
基于faster rcnn的目标检测方法，所需的计算太多，太慢，不能用于实时的目标检测  
faster rcnn:7FPS  mAP:73.2%  VOC2007test  
YOLO:45FPS mAP:63.4%  VOC2007test  
SSD:59FPS  mAP:74.3% VOC2007test  
SSD:不为假设的bounging box重采样像素或者特征  ,不产生候选域  
提升：使用小的滤波器预测物体的类别和bounding box位置的偏置  
     使用不同的预测滤波器用于不同纵横比的预测  
	 把滤波器用于网络不同的层得到的fature map去做多尺度的预测  
# 2.SSD #
介绍网络框架和训练方法  
## 模型 ##
缩短的基本网络+附加网络  
附加网络的作用：  
1. 多尺度feature map用于检测：基本网络后面加了卷积特征层，减少了尺度的增加和实现了多尺度的预测，每个特征层的预测不同， 这些增加的卷积层的 feature map 的大小变化比较大，允许能够检测出不同尺度下的物体： 在低层的feature map,感受野比较小，高层的感受野比较大，在不同的feature map进行卷积，可以达到多尺度的目的  
2. 用于检测的卷积预测器：新增加的每个卷积层的 feature map 都会通过一些小的卷积核操作，得到每一个 default boxes 关于物体类别的21个置信度 \\(c\_1,c\_2 ,\cdots, c\_p 20个类别和1个背景\\) 和4偏移 (shape offsets) 。  
3. 默认box和纵横比： 假如feature map 的size 为 m\*n, 通道数为 p，使用的卷积核大小为 3\*3\*p。每个 feature map 上的每个特征点对应 k 个 default boxes，物体的类别得分数为 c，box偏置为：4那么一个feature map就需要使用 k(c+4)个这样的卷积滤波器，最后有 (m\*n) \*k\* (c+4)个输出。默认Box跟fster rcnn中的anchor比较相似，但是默认Box是用于不同分辨率的不同feature map上的  

## 训练 ##
包含用于检测的默认box的尺度的选择和hard negative mining和数据增广策略  
**匹配策略：**需要决定那一个默认box是跟真实的Box一致的,  
对每个真实box,使用不同位置和尺度，纵横比的Box跟真实box的jaccard overlap（就是IoU），找一个最好的    
然后把默认box和任意的真实box匹配，jaccard overlap大于0.5的就选为正样本  
**训练目标：**目标函数，和常见的 Object Detection 的方法目标函数相同，分为两部分：计算相应的 default box 与目标类别的 score(置信度)以及相应的回归结果（位置回归）。置信度是采用 Softmax Loss（Faster R-CNN是log loss），位置回归则是采用 Smooth L1 loss ：  
\\(L(x, c, l, g) =\frac{1}{N}(L\_{conf} (x, c) +\alpha L\_{loc}(x,l,g)\\)N代表正样本的数目    
**为默认box选择尺度和纵横比：**  
在m的feature map上，尺度的选择计算方法：  
\\(s\_{k }= s\_{min} +\frac{s\_{max}-s\_{min}}{m -1}(k - 1); k \in [1,m]\\)  
\\(s\_{min}=0.2(最底层),s\_{max}=0.9（最高层）\\)   
纵横比的值为：\\(a\_{r}\in\\{1,2, 3,\frac{1}{2},\frac{ 1}{3}\\}\\)    
\\(w\_a^k = s\_k \sqrt{a\_r},h\_a^k = s\_k \sqrt{a\_r}\\)  
另外对于 ratio = 1 的情况，额外再指定 scale 为\\(s\_k^{'} = \sqrt{s\_ks\_{k+1}}\\) 也就是总共有 6 中不同的 default box。  
default box中心：上每个 default box的中心位置设置成 \\( ( \frac{i+0.5}{  \left| f_k \right| },\frac{j+0.5}{\left| f\_k \right| }  ) \\)，其中\\( \left| f\_k \right| \\)表示第k个特征图的大小\\( i,j \in [0, \left| f\_k \right| )\\)  
**Hard Negative Mining：**  
用于预测的 feature map 上的每个点都对应有 6 个不同的 default box，绝大部分的 default box 都是负样本，导致了正负样本不平衡。在训练过程中，采用了 Hard Negative Mining 的策略（根据confidence loss对所有的box进行排序，使正负例的比例保持在1:3） 来平衡正负样本的比率。   
**数据增广：**  
为了模型更加鲁棒，需要使用不同尺寸的输入和形状，作者对数据进行了如下方式的随机采样：  


- 使用整张图片  
- 使用IOU和目标物体为0.1, 0.3，0.5, 0.7, 0.9的patch （这些 patch 在原图的大小的 [0.1,1] 之间， 相应的宽高比在[1/2,2]之间）
- 随机采取一个patch  
当 ground truth box 的 中心（center）在采样的 patch 中时，我们保留重叠部分。在这些采样步骤之后，每一个采样的 patch 被 resize 到固定的大小，并且以 0.5 的概率随机的 水平翻转（horizontally flipped）。  
# 3.实验结果 #
## PASCAL VOC2007##
conv4_ 3,conv7 (fc7), conv8_ 2, conv9 _2, conv10 _2, and conv11_ 2预测位置和信度。  
conv4_3的尺度是0.1,conv4_3,conv10_2,conv11_2在feature map的每个位置只使用4个默认box，省略了纵横比（1/3,3）,所有的其他层使用6个默认Box    
因为conv4_3的特征尺度跟别的层不一样，所以使用L2规范化方法把conv4_3的feature map在每个位置的特征的模统一成20，这个尺度是在反向传播时学习到的  
在VOC2007上的实验结果：  
   ![](https://i.imgur.com/fQZPAhQ.png)                                                                                                                                                                 
  SSD的的定位错误很少，跟相似类别容易混淆，尤其是动物 ，在小的物体上的表现去在大的物体上表现差，提升小物体的检测准确率的空间很大   
## 模型分析 ##
![](https://i.imgur.com/kgLE9mk.png)  
数据增广很重要：跟YOLO的数据增广方法比较类似，数据增广可以提高mAP8.8%，faster rcnn的池化特征层对物体的翻转比较鲁棒  
更多的默认Box形状更好：默认的是在每个位置使用6个默认Box,移除（3，1/3）性能下降0.6%，再移除（2，1/2）性能下降2.1%  
atrous更快：本文将 VGG 中的 FC6 layer、FC7 layer 转成为 卷积层，并从模型的 FC6、FC7 上的参数，进行采样得到这两个卷积层的 parameters。还将 Pool5 layer 的参数，从 2×2−s2 转变成 3×3−s1，外加一个 pad（1），如下图：   
![](https://i.imgur.com/AmeZNoI.png)  
但是这样变化后，会改变感受野（receptive field）的大小。因此，采用了 atrous algorithm 的技术，这里所谓的 atrous algorithm，我查阅了资料，就是 hole filling algorithm。  
不同分辨率上的多个输出层更好：在不同的输出层使用不同尺度的默认Box,为了比较，每次移除一个层，但保持Box的数量不变，从74.3%递减到62.4%  
fast rcnn的ROI池化有collapsing bins问题
## PASCAL VOC2012 ##
![](https://i.imgur.com/YGixejS.png)  
## COCO ##
COCO中的物体比VOC中的物体小，所以对所有层都是用更小的默认box,最小的尺度是：0.15  
对大的物体的提升更高AP (4.8%) and AR (4.6%)，小的物体提升少AP (1.3%) and AR (2.0%)  
Faster RCNN在小物体检测上表现的比较好，原因可能是：有两个阶段的候选域筛选，RPN和fast RCNN中都对候选域进行了筛选  
![](https://i.imgur.com/4qtQxOu.png)

![](https://i.imgur.com/auxtT8K.png)  
像这种把人的手检测成人的，是正确的吗？
## 数据增广对小的物体检测的影响 ##
数据增广策略：随机裁剪可以看成放大操作，缩小操作  
需要两倍的训练时间，但mAP可以提高2%~3%  
另一个改善SSD的方法就是，产生更好的默认box,更好的跟feature map上每个位置的感受野对齐  
![](https://i.imgur.com/q75YdMJ.png)  
## 时间 ##
先用信度阈值0.01过滤掉大多数的box,然后使用对每个类jaccard overlap 0.45的NMS,为每张图片保留前200个预测  
使用的设备：batch size 8 using Titan X and cuDNN v4 with Intel Xeon E5-2667v3@3.20GHz  
![](https://i.imgur.com/NxLzcfr.png)  
# 4.相关工作 #
两大类方法：基于滑动窗的和基于候选域分类的 


----------


----------
  
SSD的方法是目前看的中，mAP和速度权衡的最好的了  
改进：改进默认box的质量，想办法提高小的物体的检测精确度
