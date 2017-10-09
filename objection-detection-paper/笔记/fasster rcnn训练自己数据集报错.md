

参考这篇博客：  
[http://blog.leanote.com/post/braveapple/%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8%E8%87%AA%E5%B7%B1%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AE%AD%E7%BB%83Faster-RCNN%E6%A8%A1%E5%9E%8B](http://blog.leanote.com/post/braveapple/%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8%E8%87%AA%E5%B7%B1%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AE%AD%E7%BB%83Faster-RCNN%E6%A8%A1%E5%9E%8B)  

----------

AttributeError: module 'google.protobuf' has no attribute 'text_format'  
[https://github.com/rbgirshick/py-faster-rcnn/issues/198](https://github.com/rbgirshick/py-faster-rcnn/issues/198)  
adding "import google.protobuf.text_format" in the train.py  


----------
NameError:name 'xrange' is not defined  
改为range

----------

AttributeError:'dict' object has no attribute 'iteritems'  
改为items


----------
pascal_voc 98行roidb=pickle.load(fid)报EOFError:   
把data/catch中的.pkl文件修改权限删除   

----------
haskey()报错，改为 in

----------
print 加括号

----------
lambda报错  
lambda (x, y): x + f(y)改为	lambda x_y: x_y[0] + f(x_y[1])  
[http://www.codexiu.cn/python/python3/39/289/](http://www.codexiu.cn/python/python3/39/289/)


----------
![](https://i.imgur.com/ctsssEr.png)
[http://blog.csdn.net/flztiii/article/details/73881954](http://blog.csdn.net/flztiii/article/details/73881954)  
只需要把fg\_rois\_per\_this\_image强制类型转换成Int,有两个地方

----------
labels[fg\_rois\_per\_this\_image:] = 0  
TypeError: slice indices must be integers or None or have an index method  
这个错误是由numpy的版本引起的，只要将fg\_rois\_per\_this\_image强制转换为int型就可以了  
labels[int(fg\_rois\_per\_this\_image):] = 0  

----------
bbox\_targets[ind, start:end] = bbox\_target\_data[ind, 1:]  
TypeError: slice indices must be integers or None or have an \__index\__ method  
解决方法：修改/py\-faster\-rcnn/lib/rpn/proposal\_target\_layer.py，转到123行  

    for ind in inds:  
        cls = clss[ind]  
        start = 4 * cls  
        end = start + 4  
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]  
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS  
    return bbox_targets, bbox_inside_weights  

这里的ind，start，end都是 numpy.int 类型，这种类型的数据不能作为索引，所以必须对其进行强制类型转换，转化结果如下：  

    for ind in inds:  
        cls = clss[ind]  
        start = 4 * cls  
        end = start + 4  
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]  
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS    
	return bbox_targets, bbox_inside_weights
minibatch.py 123行也需要这样修改

----------
self.bbox_stds[:, np.newaxis ] ValueError: operands could not be broadcast together with shapes  
修改 /py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/stage1_fast_rcnn_train.pt文件  
/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/stage1_rpn_train.pt文件
第11行    
/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/stage2_fast_rcnn_train.pt文件
第14行第380和第399行   
./py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/stage2_rpn_train.pt文件
第11行  
输入改为2，输出改为8

----------
![](https://i.imgur.com/3064vSt.png)  
删除oputput 和cache重来就行 ，原因是以前没改opt文件  

----------
FileNotFoundError: [Errno 2] No such file or directory: '/home/gj/py-faster-rcnn/data/VOCdevkit2007/results/VOC2007/Main/comp4_5158749a-5ad9-4fe4-9a90-0c260453bd00_det_test_aircraft.txt'
/data/VOCdevkit2007/目录不完整

----------
self.bbox_stds 报错 ValueError: operands could not be broadcast together with shapes   
把 with open(cachefile,'b') as f 改为 with open(cachefile,'rb') as f  
[https://github.com/endernewton/tf-faster-rcnn/issues/171](https://github.com/endernewton/tf-faster-rcnn/issues/171) 

----------
imdb.append_flipped_images()   File "/home/gj/py-faster-rcnn-ship/tools/../lib/datasets/imdb.py", line 113, in append_flipped_images     assert (boxes[:, 2] >= boxes[:, 0]).all() AssertionError
[http://blog.csdn.net/xzzppp/article/details/52036794](http://blog.csdn.net/xzzppp/article/details/52036794)   
删除cache

