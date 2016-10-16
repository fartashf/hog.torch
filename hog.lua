local C = hog.C

local HOG, parent = torch.class('hog.HOG', 'nn.Module')

function HOG:__init(sbin)
   parent.__init(self)

   self.sbin = sbin or 8

   self.grad_v = torch.CudaTensor()
   self.grad_i = torch.CudaTensor()
   self.hist = torch.CudaTensor()
   self.norm = torch.CudaTensor()
   self.output = torch.CudaTensor()
   self.gradInput = torch.CudaTensor()
end

function HOG:updateOutput(input)
   hog.typecheck(input)
   C['HOGForward'](cutorch.getState(), input:cdata(), self.output:cdata(),
       self.grad_v:cdata(), self.grad_i:cdata(), self.hist:cdata(),
       self.norm:cdata(), self.sbin)
   return self.output
end
