#include "hog.h"
#include "common.h"

// small value, used to avoid division by zero
#define eps 0.0001

/* This code is b * c * h * w. To index c, multiply by h*w */

// dims[0]=height;
// dims[1]=width;
// visible[0]=visible_height;
// visible[1]=visible_width;
// blocks[0]=blocks_height;
// blocks[1]=blocks_width;
// out[0]=hog_height;
// out[1]=hog_width;
template <typename Dtype>
__global__ void GradForward(const int nthreads, const Dtype* im, const int num,
        const int channels, const int height, const int width,
        const int hist_channels, const int blocks_height, const int
        blocks_width, const int visible_height, const int  visible_width,
        const int out_channels, const int out_height, const int out_width,
        const int sbin,
        Dtype* grad_v, Dtype* grad_i, Dtype* hist, Dtype* norm, Dtype* feat){
    // unit vectors used to compute gradient orientation
    Dtype uu[9] = {1.0000, 
        0.9397, 
        0.7660, 
        0.500, 
        0.1736, 
        -0.1736, 
        -0.5000, 
        -0.7660, 
        -0.9397};
    Dtype vv[9] = {0.0000, 
        0.3420, 
        0.6428, 
        0.8660, 
        0.9848, 
        0.9848, 
        0.8660, 
        0.6428, 
        0.3420};

    CUDA_KERNEL_LOOP(index, nthreads) {
        int w = index % (visible_width-2) + 1;
        int h = (index / (visible_width-2)) % (visible_height-2) + 1;
        int n = (index / (visible_width-2)) / (visible_height-2);
        int pw = min(w, width-2);
        int ph = min(h, height-2);

        // first color channel
        const Dtype* im_off = im + (n*channels*height + ph) * width + pw;
        Dtype dy = im_off[width] - im_off[-width];
        Dtype dx = im_off[1] - im_off[-1];
        Dtype v = dx*dx + dy*dy;

        // second color channel
        im_off += height * width;
        Dtype dy2 = im_off[width] - im_off[-width];
        Dtype dx2 = im_off[1] - im_off[-1];
        Dtype v2 = dx2*dx2 + dy2*dy2;

        // third color channel
        im_off += height * width;
        Dtype dy3 = im_off[width] - im_off[-width];
        Dtype dx3 = im_off[1] - im_off[-1];
        Dtype v3 = dx3*dx3 + dy3*dy3;

        // pick channel with strongest gradient
        if (v2 > v) {
            v = v2;
            dx = dx2;
            dy = dy2;
        } 
        if (v3 > v) {
            v = v3;
            dx = dx3;
            dy = dy3;
        }

        // snap to one of 18 orientations
        Dtype best_dot = 0;
        int best_o = 0;
        for (int o = 0; o < 9; o++) {
            Dtype dot = uu[o]*dx + vv[o]*dy;
            if (dot > best_dot) {
                best_dot = dot;
                best_o = o;
            } else if (-dot > best_dot) {
                best_dot = -dot;
                best_o = o+9;
            }
        }
        v = sqrt(v);

        grad_v[index] = v;
        grad_i[index] = best_o;
    }
}

template <typename Dtype>
__global__ void BinForward(const int nthreads, const Dtype* im, const int num,
        const int channels, const int height, const int width,
        const int hist_channels, const int blocks_height, const int
        blocks_width, const int visible_height, const int  visible_width,
        const int out_channels, const int out_height, const int out_width,
        const int sbin,
        Dtype* grad_v, Dtype* grad_i, Dtype* hist, Dtype*norm, Dtype*feat){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int wb = index % blocks_width;
        int hb = (index / blocks_width) % blocks_height;
        int ob = ((index / blocks_width) / blocks_height) % hist_channels;
        int n = ((index / blocks_width) / blocks_height) / hist_channels;

        // add to 4 histograms around pixel using linear interpolation
        int w0 = (int)floor(((Dtype)wb-1+0.5)*(Dtype)sbin -0.5);
        int w1 = (int)ceil(((Dtype)wb+1+0.5)*(Dtype)sbin -0.5);
        int h0 = (int)floor(((Dtype)hb-1+0.5)*(Dtype)sbin -0.5);
        int h1 = (int)ceil(((Dtype)hb+1+0.5)*(Dtype)sbin -0.5);
        Dtype sum = 0.0;
        // for (int w = 1; w < visible_width-1; w++) {
        //     for(int h = 1; h < visible_height-1; h++) {
        for (int w = max(1, w0); w < min(visible_width-1, w1); w++){
            for (int h = max(1, h0); h < min(visible_height-1, h1); h++){
                Dtype wp = ((Dtype)w+0.5)/(Dtype)sbin - 0.5;
                Dtype hp = ((Dtype)h+0.5)/(Dtype)sbin - 0.5;
                int iwp = (int)floor(wp);
                int ihp = (int)floor(hp);
                Dtype vw0 = wp-iwp;
                Dtype vh0 = hp-ihp;
                Dtype vw1 = 1.0-vw0;
                Dtype vh1 = 1.0-vh0;
                int iw = w-1;
                int ih = h-1;
                int o = grad_i[(n*(visible_height-2)+ih)*(visible_width-2)+iw];
                Dtype v = grad_v[(n*(visible_height-2)+ih)*(visible_width-2)+iw];
                if (iwp == wb && ihp == hb && o == ob) {
                    sum += vw1*vh1*v;
                }
                if (iwp+1 == wb && ihp == hb && o == ob) {
                    sum += vw0*vh1*v;
                }
                if (iwp == wb && ihp+1 == hb && o == ob) {
                    sum += vw1*vh0*v;
                }
                if (iwp+1 == wb && ihp+1 == hb && o == ob) {
                    sum += vw0*vh0*v;
                }
            }
        }
        hist[index] = sum;
    }
}

template <typename Dtype>
__global__ void NormForward(const int nthreads, const Dtype* im, const int num,
        const int channels, const int height, const int width,
        const int hist_channels, const int blocks_height, const int
        blocks_width, const int visible_height, const int  visible_width,
        const int out_channels, const int out_height, const int out_width,
        const int sbin,
        Dtype* grad_v, Dtype* grad_i, Dtype* hist, Dtype*norm, Dtype*feat){

    CUDA_KERNEL_LOOP(index, nthreads) {
        int w = index % blocks_width;
        int h = (index / blocks_width) % blocks_height;
        int n = (index / blocks_width) / blocks_height;
        Dtype sum = 0.0;
        for (int o = 0; o < 9; o++) {
            int off1 = ((n*hist_channels + o)*blocks_height + h)*blocks_width + w;
            int off2 = ((n*hist_channels + (o+9))*blocks_height + h)*blocks_width + w;
            sum += (hist[off1]+hist[off2])*(hist[off1]+hist[off2]);
        }
        norm[index] = sum;
    }
}

template <typename Dtype>
__global__ void CompForward(const int nthreads, const Dtype* im, const int num,
        const int channels, const int height, const int width,
        const int hist_channels, const int blocks_height, const int
        blocks_width, const int visible_height, const int  visible_width,
        const int out_channels, const int out_height, const int out_width,
        const int sbin,
        Dtype* grad_v, Dtype* grad_i, Dtype* hist, Dtype*norm, Dtype*feat){

    CUDA_KERNEL_LOOP(index, nthreads) {
        int w = index % out_width;
        int h = (index / out_width) % out_height;
        int n = (index / out_width) / out_height;
        int off;
        Dtype n1, n2, n3, n4;
        off = (n*blocks_height + (h+1))*blocks_width + (w+1);
        n1 = 1.0 / sqrt(norm[off] + norm[off+1] + norm[off+blocks_width]
                + norm[off+blocks_width+1] + eps);
        off = (n*blocks_height + h)*blocks_width + (w+1);
        n2 = 1.0 / sqrt(norm[off] + norm[off+1] + norm[off+blocks_width]
                + norm[off+blocks_width+1] + eps);
        off = (n*blocks_height + (h+1))*blocks_width + w;
        n3 = 1.0 / sqrt(norm[off] + norm[off+1] + norm[off+blocks_width]
                + norm[off+blocks_width+1] + eps);
        off = (n*blocks_height + h)*blocks_width + w;
        n4 = 1.0 / sqrt(norm[off] + norm[off+1] + norm[off+blocks_width]
                + norm[off+blocks_width+1] + eps);

        Dtype t1 = 0;
        Dtype t2 = 0;
        Dtype t3 = 0;
        Dtype t4 = 0;

        // contrast-sensitive features
        int hoff = (n*hist_channels*blocks_height + (h+1))*blocks_width + (w+1);
        int foff = (n*out_channels*out_height + h)*out_width + w;
        for (int o = 0; o < 18; o++) {
            Dtype h1 = min(hist[hoff] * n1, 0.2);
            Dtype h2 = min(hist[hoff] * n2, 0.2);
            Dtype h3 = min(hist[hoff] * n3, 0.2);
            Dtype h4 = min(hist[hoff] * n4, 0.2);
            feat[foff] = 0.5 * (h1 + h2 + h3 + h4);
            t1 += h1;
            t2 += h2;
            t3 += h3;
            t4 += h4;
            foff += out_height*out_width;
            hoff += blocks_height*blocks_width;
        }

        // contrast-insensitive features
        hoff = (n*hist_channels*blocks_height + (h+1))*blocks_width + (w+1);
        for (int o = 0; o < 9; o++) {
            Dtype sum = hist[hoff] + hist[hoff+9*blocks_width*blocks_height];
            Dtype h1 = min(sum * n1, 0.2);
            Dtype h2 = min(sum * n2, 0.2);
            Dtype h3 = min(sum * n3, 0.2);
            Dtype h4 = min(sum * n4, 0.2);
            feat[foff]= 0.5 * (h1 + h2 + h3 + h4);
            foff += out_height*out_width;
            hoff += blocks_height*blocks_width;
        }

        // texture features
        feat[foff] = 0.2357 * t1;
        foff += out_height*out_width;
        feat[foff] = 0.2357 * t2;
        foff += out_height*out_width;
        feat[foff] = 0.2357 * t3;
        foff += out_height*out_width;
        feat[foff] = 0.2357 * t4;
    }
}


void HOGForward(THCState *state,
        THCudaTensor *input, THCudaTensor *output, THCudaTensor *grad_v,
        THCudaTensor *grad_i, THCudaTensor *hist,
        THCudaTensor *norm, int sbin)
{
    THCUNN_assertSameGPU(state, 4, input, output, hist, norm);
    THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2,
            "3D or 4D (batch) tensor expected");

    long nInputCols, nInputRows, nInputPlane, batchSize;
    long nOutputCols, nOutputRows, nOutputPlane;

    if (input->nDimension == 3) {
        nInputCols = input->size[2];
        nInputRows = input->size[1];
        nInputPlane = input->size[0];
        batchSize = 1;
    }
    else
    {
        nInputCols = input->size[3];
        nInputRows = input->size[2];
        nInputPlane = input->size[1];
        batchSize = input->size[0];
    }

  if(nInputPlane != 3)
      THError("Given input size: (%dx%dx%d). Expected 3 input channels",
              nInputPlane,nInputRows,nInputCols);

  // memory for caching orientation histograms & their norms
  long blocks_height = (long)round((float)nInputRows/(float)sbin);
  long blocks_width = (long)round((float)nInputCols/(float)sbin);
  long visible_height = blocks_height*sbin;
  long visible_width = blocks_width*sbin;

  THCudaTensor_resize3d(state, grad_v, batchSize, visible_height-2,
          visible_width-2);
  THCudaTensor_resize3d(state, grad_i, batchSize, visible_height-2,
          visible_width-2);
  long hist_channels = 18;
  THCudaTensor_resize4d(state, hist, batchSize, hist_channels, blocks_height, blocks_width);
  THCudaTensor_resize3d(state, norm, batchSize, blocks_height, blocks_width);

  // memory for HOG features
  nOutputRows = max(blocks_height-2, 0l);
  nOutputCols = max(blocks_width-2, 0l);
  nOutputPlane = 27+4;
  THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, nOutputRows,
          nOutputCols);
  
  input = THCudaTensor_newContiguous(state, input);
  float* input_data = THCudaTensor_data(state, input);
  float* grad_v_data = THCudaTensor_data(state, grad_v);
  float* grad_i_data = THCudaTensor_data(state, grad_i);
  float* hist_data = THCudaTensor_data(state, hist);
  float* norm_data = THCudaTensor_data(state, norm);
  float* output_data = THCudaTensor_data(state, output);

  int count = batchSize*(visible_height-2)*(visible_width-2);
  GradForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0,
              THCState_getCurrentStream(state) >>>
                  (count, input_data, batchSize,
                   nInputPlane, nInputRows, nInputCols,
                   hist_channels, blocks_height, blocks_width, 
                   visible_height, visible_width, nOutputPlane, nOutputRows,
                   nOutputCols, sbin,
                   grad_v_data, grad_i_data, hist_data, norm_data, output_data);
  THCudaCheck(cudaGetLastError());

  count = THCudaTensor_nElement(state, hist);
  BinForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0,
              THCState_getCurrentStream(state) >>>
                  (count, input_data, batchSize,
                   nInputPlane, nInputRows, nInputCols,
                   hist_channels, blocks_height, blocks_width, 
                   visible_height, visible_width, nOutputPlane, nOutputRows,
                   nOutputCols, sbin,
                   grad_v_data, grad_i_data, hist_data, norm_data, output_data);
  THCudaCheck(cudaGetLastError());

  count = batchSize*blocks_width*blocks_height;
  NormForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0,
              THCState_getCurrentStream(state) >>>
                  (count, input_data, batchSize,
                   nInputPlane, nInputRows, nInputCols,
                   hist_channels, blocks_height, blocks_width, 
                   visible_height, visible_width, nOutputPlane, nOutputRows,
                   nOutputCols, sbin,
                   grad_v_data, grad_i_data, hist_data, norm_data, output_data);
  THCudaCheck(cudaGetLastError());

  count = batchSize*nOutputRows*nOutputCols;
  CompForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0,
              THCState_getCurrentStream(state) >>>
                  (count, input_data, batchSize,
                   nInputPlane, nInputRows, nInputCols,
                   hist_channels, blocks_height, blocks_width, 
                   visible_height, visible_width, nOutputPlane, nOutputRows,
                   nOutputCols, sbin,
                   grad_v_data, grad_i_data, hist_data, norm_data, output_data);
  THCudaCheck(cudaGetLastError());

  if(input->nDimension == 3)
      THCudaTensor_resize3d(state, output, nOutputPlane, nOutputRows,
              nOutputCols);

  THCudaTensor_free(state, input);
}
