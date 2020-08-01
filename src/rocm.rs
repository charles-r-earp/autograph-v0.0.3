use std::{sync::{Arc, Mutex, LockResult, MutexGuard, PoisonError}, ffi::{CString, c_void}};

use hip_sys::hiprt::hipError_t;
use hip_sys::hipblas::{
    hipblasStatus_t,
    hipblasHandle_t,
    hipblasCreate,
    hipblasDestroy,
    hipblasSetStream,
};
use miopen_sys::{
    miopenStatus_t,
    miopenHandle_t,
    miopenCreateWithStream,
    miopenDestroy
};

mod error;
use error::{RocmError, RocmResult, IntoResult};

#[doc(hidden)]
#[macro_use]
pub mod rustacuda_like;
pub(crate) use rustacuda_like::DeviceCopy;
use rustacuda_like::{RocmDevice, Context, ContextFlags, CurrentContext, Stream, StreamFlags, Module, DevicePointer, DeviceBuffer, DeviceSlice}; 

#[doc(hidden)]
pub struct Hipblas {
    handle: hipblasHandle_t                        
}

impl Hipblas {
    fn with_stream(stream: &Stream) -> RocmResult<Self> {
        let mut handle: hipblasHandle_t = std::ptr::null_mut();
        let status = unsafe {
            hipblasCreate(
                &mut handle as *mut hipblasHandle_t
            )
        };
        status.into_result()?;
        let status = unsafe {
            hipblasSetStream(
                handle,
                stream.as_mut_ptr() as hip_sys::hipblas::hipStream_t
            )
        };
        status.into_result()?;
        Ok(Self { handle })
    }
    unsafe fn as_mut_ptr(&self) -> hipblasHandle_t {
        self.handle
    }
}

impl Drop for Hipblas {
    fn drop(&mut self) {
        let status = unsafe {
            hipblasDestroy(self.handle)
        };
        status.into_result()
            .unwrap(); 
    }
}

#[doc(hidden)]
pub struct Miopen {
    handle: miopenHandle_t
}

impl Miopen {
    fn with_stream(stream: &Stream) -> RocmResult<Self> {
        let mut handle: miopenHandle_t = std::ptr::null_mut();
        let status = unsafe {
            miopenCreateWithStream(
                &mut handle as *mut miopenHandle_t,
                stream.as_mut_ptr() as miopen_sys::hipStream_t
            )
        };
        status.into_result()?;
        Ok(Self { handle })     
    }
    unsafe fn as_mut_ptr(&self) -> miopenHandle_t {
        self.handle
    }
}

#[doc(hidden)]
pub struct RocmGpuBase {
    stream: Stream,
    kernels: Module,
    hipblas: Hipblas,
    miopen: Miopen,
    context: Context
}

impl RocmGpuBase {
    fn stream(&self) -> &Stream { 
        &self.stream
    }
    fn kernels(&self) -> &Module {
        &self.kernels
    }
    fn blas(&self) -> &Hipblas {
        &self.hipblas
    }
    fn nn(&self) -> &Miopen {
        &self.miopen
    }
    fn context(&self) -> &Context {
        &self.context 
    }
}

/// Safe wrapper for several ROCm implementation handles
pub struct RocmGpu {
    index: usize,
    device: RocmDevice,
    base: Mutex<RocmGpuBase>
}

impl RocmGpu {
    pub fn new(index: usize) -> Arc<Self> {    
        let device = RocmDevice::get_device(index as u32)
            .expect(&format!("RocmGpu unable to get device {}!", index));
        let context = Context::create_and_push(ContextFlags::empty(), device)
            .expect("Unable to create Rocm Context!");
        let stream = Stream::new(StreamFlags::empty(), Some(0))
            .expect("Unable to create Rocm Stream!");
        let src = CString::new(include_str!("rocm/kernels.s")).unwrap();
        let kernels = Module::load_from_string(&src).unwrap();
        let hipblas = Hipblas::with_stream(&stream)
            .expect("Unable to create Hipblas!");
        let miopen = Miopen::with_stream(&stream)
            .expect("Unable to create Miopen!");
        let base = Mutex::new(RocmGpuBase {
            stream,
            kernels,
            hipblas,
            miopen,
            context
        });
        Arc::new(Self {
            index,
            device,
            base
        })
    }
    fn lock(&self) -> LockResult<MutexGuard<RocmGpuBase>> {
        self.base.lock()
            .map(|base| {
                CurrentContext::set_current(base.context())
                    .expect("Unable to set CurrentContext!");
                base
            })
            .map_err(|e| {
                let base = e.into_inner();
                CurrentContext::set_current(base.context())
                    .expect("Unable to set CurrentContext!");
                PoisonError::new(base)
            })
    }
    pub(super) fn synchronize(&self) {
        self.lock()
            .expect("Unable to lock RocmGpu!")
            .stream()
            .synchronize()
            .expect("Unable to synchronize Rocm Stream!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_rocm_gpu_new() {
        RocmGpu::new(0);
    }
}
