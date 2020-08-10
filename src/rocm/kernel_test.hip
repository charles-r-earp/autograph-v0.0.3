#include <hip/hip_runtime.h>

extern "C" {
    __global__ void ones(float* y) {
        y[0] = 1.;
    }
}
