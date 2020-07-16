
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
        if (x[tid] > 0.0f) {
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

__kernel void conv2d_broadcast_bias_backward(__global float* db, __global const float* dy, unsigned int nchannels, unsigned int filter_len, unsigned int len) {
    size_t tid = get_global_id(0);
    if (tid < len) {
        size_t batch = tid / nchannels;
        size_t channel = tid % nchannels;
        for(int i = 0; i < filter_len; ++i) {
            db[channel] += dy[batch*nchannels*filter_len + channel*filter_len + i];
        }
    }
}
  
__kernel void max_pool2d_forward(__global const float* x, __global float* y, 
                                  unsigned int ih, unsigned int iw,
                                  unsigned int oh, unsigned int ow,
                                  unsigned int kh, unsigned int kw, 
                                  unsigned int sh, unsigned int sw, 
                                  unsigned int ph, unsigned int pw,
                                  unsigned int len) {
    size_t tid = get_global_id(0);
    size_t y_idx, x_idx;
    int iy, ix; 
    if (tid < len) {
        iy = -ph;
        for(int oy = 0; iy < oh; ++oy) {
            ix = -pw;
            for(int ox = 0; ox < ow; ++ox) {
                y_idx = tid*oh*ow + oy*ow + ox;
                for(int ky = 0; ky < kh; ++ky) {
                    if (iy + ky >= 0 && iy + ky < ih) {
                        for(int kx = 0; kx < kw; ++kx) {
                            x_idx = tid*ih*iw + (iy+ky)*iw + (ix+kx);
                            if (ky == 0 && kx == 0) {
                                y[y_idx] = x[x_idx];
                            }
                            else if (ix + kx >= 0 && ix + kx < iw) {
                                if (x[x_idx] > y[y_idx]) {
                                    y[y_idx] = x[x_idx];
                                }
                            }
                        } 
                    }
                }
                ix += sw;
            }
            iy += sh; 
        }    
    }
}

__kernel void max_pool2d_backward(__global const float* x, __global float* dx, __global const float* dy, 
                                  unsigned int ih, unsigned int iw,
                                  unsigned int oh, unsigned int ow,
                                  unsigned int kh, unsigned int kw, 
                                  unsigned int sh, unsigned int sw, 
                                  unsigned int ph, unsigned int pw,
                                  unsigned int len) {
    size_t tid = get_global_id(0);
    if (tid < len) {
        size_t y_idx, x_idx, max_idx;
        int iy = -ph, ix;
        for(int oy = 0; iy < oh; ++oy) {
            ix = -pw;
            for(int ox = 0; ox < ow; ++ox) {
                y_idx = tid*oh*ow + oy*ow + ox;
                for(int ky = 0; ky < kh; ++ky) {
                    if (iy + ky >= 0 && iy + ky < ih) {
                        for(int kx = 0; kx < kw; ++kx) {
                            x_idx = tid*ih*iw + (iy+ky)*iw + (ix+kx);
                            if (ky == 0 && kx == 0) {
                                max_idx = x_idx;
                            }
                            else if (ix + kx >= 0 && ix + kx < iw) {
                                if (x[x_idx] > x[max_idx]) {
                                    max_idx = x_idx;
                                }
                            }
                        } 
                    }
                }
                dx[max_idx] += dy[y_idx];
                ix += sw;
            }
            iy += sh; 
        }    
    }
}
/*
__kernel void nchw_to_nhwc(__global const float* x, __global float* y,
                           const unsigned int batch_size, const unsigned int nchannels, 
                           const unsigned int h, const unsigned int w, 
                           const unsigned int len) {
    size_t tid = get_global_id(0);
    if (tid < len) {
        size_t n = tid / nchannels;
        size_t c = tid % nchannels;
        for(int i = 0; i < h; ++i) {
            for(int j = 0; j < w; ++j) {
                y[n*h*w*nchannels + i*w*nchannels + j*nchannels + c] = x[n*nchannels*h*w + c*h*w + i*w + j];
            }
        }
    }
}

__kernel void nhwc_to_nchw(__global const float* x, __global float* y, const float beta,
                           const unsigned int batch_size, const unsigned int nchannels, 
                           const unsigned int h, const unsigned int w, 
                           const unsigned int len) {
    size_t tid = get_global_id(0);
    size_t x_idx, y_idx;
    if (tid < len) {
        size_t n = tid / nchannels;
        size_t c = tid % nchannels;
        for(int i = 0; i < h; ++i) {
            for(int j = 0; j < w; ++j) {
                x_idx = n*h*w*nchannels + i*w*nchannels + j*nchannels + c;
                y_idx = n*nchannels*h*w + c*h*w + i*w + j;
                y[y_idx] = x[x_idx] + beta * y[y_idx];
            }
        }
    }
}

__kernel void ohwi_flipped_to_oihw(__global const float* x, __global float* y, const float beta,
                           const unsigned int outputs, const unsigned int inputs, 
                           const unsigned int kh, const unsigned int kw, 
                           const unsigned int len) {
    size_t tid = get_global_id(0);
    size_t x_idx, y_idx;
    if (tid < len) {
        size_t out = tid / inputs;
        size_t in = tid % inputs;
        for(int i = 0; i < kh; ++i) {
            for(int j = 0; j < kw; ++j) {
                x_idx = out*kh*kw*inputs + (kh-i)*kw*inputs + (kw-j)*inputs + in;
                y_idx = out*inputs*kh*kw + in*kh*kw + i*kw + j;
                y[y_idx] = x[x_idx] + beta * y[y_idx];
            }
        }
    }
}*/
