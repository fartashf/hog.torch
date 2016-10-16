hog.torch
====================

A Cuda implementation of Histogram of Oriented Gradients (HOG) from
[`voc-release5`](https://github.com/rbgirshick/voc-dpm/blob/master/features/features.cc).

#### Installation
`luarocks make`


#### Usage
```lua
require 'hog';
im = image.load('image.jpg')
h = hog.HOG()
im_cuda = torch.CudaTensor()
im_cuda:resize(im:size())
im_cuda:copy(im)
feat_gpu = h:forward(im_cuda)
cutorch.synchronize()
feat = feat_gpu:float()
```
