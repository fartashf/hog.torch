-- from cuda-convnet2
local ffi = require 'ffi'

ffi.cdef[[
void HOGForward(THCState *state,
        THCudaTensor *input, THCudaTensor *output, THCudaTensor *grad_v,
        THCudaTensor *grad_i, THCudaTensor *hist,
        THCudaTensor *norm, int sbin);
]]

local path = package.searchpath('libhog', package.cpath)
if not path then
   path = require 'hog.config'
end
assert(path, 'could not find libhog.so')
hog.C = ffi.load(path)
