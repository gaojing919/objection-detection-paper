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
2. 用于检测的卷积预测器：新增加的每个卷积层的 feature map 都会通过一些小的卷积核操作，得到每一个 default boxes 关于物体类别的21个置信度 \\(c_1,c_2 ,\cdots, c_p 20个类别和1个背景\\) 和4偏移 (shape offsets) 。  
3. 默认box和纵横比： 假如feature map 的size 为 m\*n, 通道数为 p，使用的卷积核大小为 3\*3\*p。每个 feature map 上的每个特征点对应 k 个 default boxes，物体的类别得分数为 c，box偏置为：4那么一个feature map就需要使用 k(c+4)个这样的卷积滤波器，最后有 (m\*n) \*k\* (c+4)个输出。默认Box跟fster rcnn中的anchor比较相似，但是默认Box是用于不同分辨率的不同feature map上的  

## 训练 ##
包含用于检测的默认box的尺度的选择和hard negative mining和数据增广策略  
**匹配策略：**需要决定那一个默认box是跟真实的Box一致的,  
对每个真实box,使用不同位置和尺度，纵横比的Box跟真实box的jaccard overlap（就是IoU），找一个最好的    
然后把默认box和任意的真实box匹配，jaccard overlap大于0.5的就选为正样本  
**训练目标：**目标函数，和常见的 Object Detection 的方法目标函数相同，分为两部分：计算相应的 default box 与目标类别的 score(置信度)以及相应的回归结果（位置回归）。置信度是采用 Softmax Loss（Faster R-CNN是log loss），位置回归则是采用 Smooth L1 loss ：  
\\(L(x, c, l, g) =\frac{1}{N}(L\_{conf} (x, c) +\alpha L_{loc}(x,l,g)\\)N代表正样本的数目    
**为默认box选择尺度和纵横比：**  
在m的feature map上，尺度的选择计算方法：  
\\(s\_{k }= s\_{min} +\frac{s\_{max}-s\_{min}}{m -1}(k - 1); k \in [1,m]\\)  
\\(s\_{min}=0.2(最底层),s\_{max}=0.9（最高层）\\)   
纵横比的值为：\\(a\_{r}\in\\{1,2, 3,\frac{1}{2},\frac{ 1}{3}\\}\\)    
\\(w_a^k = s_k\sqrt{a_r},h_a^k = s_k/\sqrt{a_r}\\)  
另外对于 ratio = 1 的情况，额外再指定 scale 为\\(s\_k^{'} = \sqrt{s\_ks\_{k+1}}\\) 也就是总共有 6 中不同的 default box。  
default box中心：上每个 default box的中心位置设置成 \\( ( \frac{i+0.5}{  \left| f_k \right| },\frac{j+0.5}{\left| f_k \right| }  ) \\)，其中\\( \left| f_k \right| \\)表示第k个特征图的大小\\( i,j \in [0, \left| f_k \right| )\\)  
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






