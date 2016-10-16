function hog.typecheck(i)
   if torch.type(i) ~= 'torch.CudaTensor' then 
      error('Input is expected to be torch.CudaTensor') 
   end
end
