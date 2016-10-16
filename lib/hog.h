#ifndef HOG_CUH
#define HOG_CUH
#include <THC/THC.h>

extern "C" {
void HOGForward(THCState *state,
        THCudaTensor *input, THCudaTensor *output, THCudaTensor *grad_v,
        THCudaTensor *grad_i, THCudaTensor *hist,
        THCudaTensor *norm, int sbin);
}
#endif  /* HOG_CUH */
