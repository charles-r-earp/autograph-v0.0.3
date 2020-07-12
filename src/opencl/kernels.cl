
 __kernel void u8_to_f32(__global const unsigned char* x, __global float* y, unsigned int len) {
    const float scale = 1.0f / 255.0f;
    int tid = get_global_id(0);
    if (tid < len) {
      y[tid] = scale * x[tid];
    }
 }
 
 __kernel void u8_to_one_hot_f32(__global const unsigned char* x, unsigned int nclasses, __global float* y, unsigned int len) {
    int tid = get_global_id(0);
    if (tid < len) {
      y[tid*nclasses+x[tid]] = 1.0f;
    }
 } 
 
 __kernel void axpy_f32(const unsigned int n, const float alpha, __global const float *x, const unsigned int incx, __global float* y, const unsigned int incy) {
    int tid = get_global_id(0);
    if (tid < n) {
        y[tid*incy] += alpha * x[tid*incx];
    }
 }
 
 __kernel void cross_entropy_forward(unsigned int batch_size, unsigned int nclasses, __global const float* x, __global const float* t, __global float* y) {
    int tid = get_global_id(0);
    if (tid < batch_size) {
      // compute max value of slice
      float m = x[tid*nclasses];
      for(int i = 1; i < nclasses; ++i) {
        m = fmax(x[tid*nclasses+i], m);
      } 
      // subtract max
      for(int i = 0; i < nclasses; ++i) {
        y[tid*nclasses+i] = x[tid*nclasses+i]-m;
      }
      // sum
      float s = 0.0f;
      for(int i = 0; i < nclasses; ++i) {
        s += exp(y[tid*nclasses+i]);
      }
      // compute ln(s)
      float ln_s = log(s);
      // y = (ln_s - y) * t
      for(int i = 0; i < nclasses; ++i) {
        y[tid*nclasses+i] = (ln_s - y[tid*nclasses+i]) * t[tid*nclasses+i];
      }
    }
  }
  
  __kernel void cross_entropy_backward(__global const float* x, __global float* dx, __global const float* t, __global float* dy, unsigned int len) {
    size_t tid = get_global_id(0);
    if (tid < len) {
      dx[tid] = dy[0] * (x[tid] - t[tid]);
    }
  }
  
  __kernel void relu_forward(__global const float* x, __global float* y, unsigned int len) {
    size_t tid = get_global_id(0);
    if (tid < len) {
        y[tid] = x[tid] >= 0.0f ? x[tid] : 0.0f;
    }
  }
  
  __kernel void relu_backward(__global const float* x, __global float* dx, __global const float* dy, unsigned int len) {
    size_t tid = get_global_id(0);
    if (tid < len) {
        if (x[tid] >= 0.0f) {
            dx[tid] += dy[tid];
        }
    }
  }
  
   __kernel void reverse_conv_filter(__global const float* x, float beta, __global float* y, unsigned int filter_len, unsigned int len) {
    size_t tid = get_global_id(0);
    if (tid < len) {
      if (beta == 0.0f) {
        for(int i = 0; i < filter_len; ++i) {
          y[tid*filter_len + i] = x[tid*filter_len + ((filter_len-1) - i)];
        }
      }
      else {
        for(int i = 0; i < filter_len; ++i) {
          y[tid*filter_len + i] = x[tid*filter_len + ((filter_len-1) - i)] + beta * y[tid*filter_len + i];
        }
      }
    }
  }
  
  __kernel void conv2d_broadcast_bias(__global const float* b, __global float* y, unsigned int nchannels, unsigned int filter_len, unsigned int len) {
    size_t tid = get_global_id(0);
    if (tid < len) {
        size_t batch = tid / nchannels;
        size_t channel = tid % nchannels;
        for(int i = 0; i < filter_len; ++i) {
            y[batch*nchannels*filter_len + channel*filter_len + i] = b[channel];
        }
    }
  }
