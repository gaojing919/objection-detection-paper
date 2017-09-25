# Mimic方法改进 #
原来的网络结构为：  
![](http://i.imgur.com/c0qa9Jt.png)  
改进后的网络结构为：  
![](http://i.imgur.com/hqkm9SH.png)  
  
原来的损失函数为：    
$$ L(W)= \lambda_1 L_m (W)+L_ {gt} \(W\) $$  
$$ L_m(W)= \frac{1}{2N}\sum_ {i}\frac{1}{m_i}||u^{(i)}-r(v^{(i)})||_ {2}^{2} $$  
$$ L_ {gt}=L_ {cls}(W)+\lambda_ {2}L_ {reg}(W) $$  
修改后的损失函数为：  
$$ L1_m(W)= \frac{1}{2N}\sum_ {i}\frac{1}{m_i}||u_1^{(i)}-r(v^{(i)})||_2^2 $$   
$$ L2_m(W)= \frac{1}{2N}\sum_ {i}\frac{1}{m_i}||u_2^{(i)}-r(v^{(i)})||_2^2 $$         
$$ L_m(W)=x_ {1}L1_m+x_ {2}L2_m $$   
如果大网络1检测到了物体\\(x_ {1}=1\\)否则等于0  
如果大网络2检测到了物体\\(x_ {2}=1\\)否则等于0   
损失函数可以理解为：如果那个大网络检测到了物体则可以理解为，这个网络对这张图片提取的feature map比较好，应该用这个网络来监督小网络训练，如果大网络没有检测到物体，那么说明这个大网络对这张图片提取的feature map不好，不应该用来误导小网络  
使用的两个大网络要仔细选择，计算特征的方法不同，但性能相差不多的网络，比如选fast rcnn和faster rcnn就不好，因为这两个网络计算的特征应该是比较像的，而且faster明显优于fast，这样的话fast的贡献就太少了



初步选定：Large CNN(1):faster rcnn  ,Large CNN(2):RFCN   
inception是一种压缩网络的方法，有V1，V2，V3，V4四个版本，作者用的是V1，所以我需要去确定网络结构选用什么？  
作者的做法是：小网络是由压缩过的大网络得到，所以我需要决定自己使用的小网络的网络结构 
比如像：1/2 inception-mimic+finetune RPN就是网络结构就是：压缩过的fast RCNN+压缩过的RPN,大网络是：faster rcnn:fast RCNN+RPN  
作者的想法可能是：网络结构一样，只是经过压缩，这样去模仿feature map会比较好
  
需要做的事：  
- 决定小网络的结构，实现小网络，用  inception方法压缩，inception V1的代码网上有  （这里代码可能比较多）  
- 小网络在image net上预训练  
- 用Large CNN(1),Large CNN(2)，真实数据联合微调小网络




# 另一个 #

不止学习最后一层产生的feature map，前面几层的也学习，分类和bounding box 回归时也用到前面几层的那个特征，有点像SSD