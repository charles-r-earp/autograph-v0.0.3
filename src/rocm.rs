use std::{sync::{Arc, Mutex}, ffi::{CStr, CString}};

use hip_sys::hiprt::{
    hipError_t,
    hipDevice_t,
    hipDeviceGet,
    hipCtx_t,
    hipCtxCreate,
    hipCtxDestroy,
    hipCtxSetCurrent,
    hipStream_t,
    hipStreamCreateWithPriority,
    hipStreamDestroy,
    hipModule_t,
    hipModuleLoadData,
};
use hip_sys::hipblas::{
    hipblasStatus_t,
    hipblasHandle_t,
    hipblasCreate,
    hipblasDestroy,
};
use miopen_sys::{
    miopenStatus_t,
    miopenHandle_t,
    miopenCreateWithStream,
    miopenDestroy
};

mod error;
use error::{RocmError, RocmResult, IntoResult};

#[derive(Clone, Copy)]
struct RocmDevice {
    device: hipDevice_t
}

impl RocmDevice {
    fn get_device(ordinal: u32) -> RocmResult<Self> {
        let mut device = hipDevice_t::default(); 
        let error = unsafe {
            hipDeviceGet(
                &mut device as *mut hipDevice_t,
                ordinal as i32
            )
        };
        error.into_result()?;
        Ok(Self{device})
    }
}

struct StreamFlags {
    bits: u32
}

impl StreamFlags {
    const DEFAULT: StreamFlags = StreamFlags { bits: 0 };
    fn empty() -> Self { 
        Self { bits: 0 }
    }
}

#[doc(hidden)]
pub struct Stream {
    stream: hipStream_t
}

impl Stream {
    fn new(flags: StreamFlags, priority: Option<i32>) -> RocmResult<Self> {
        let mut stream: hipStream_t = std::ptr::null_mut();
        let error = unsafe {
            hipStreamCreateWithPriority(
                &mut stream as *mut hipStream_t,
                flags.bits,
                priority.unwrap_or(0)
            )
        };
        error.into_result()?;
        Ok(Self { stream })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let error = unsafe {
            hipStreamDestroy(self.stream)
        };
        error.into_result()
            .unwrap();
    }
}

#[doc(hidden)]
pub struct Module {
    module: hipModule_t
}

impl Module {
    fn load_from_string(image: &CStr) -> RocmResult<Self> {
        let mut module: hipModule_t = std::ptr::null_mut();
        let error = unsafe {
            hipModuleLoadData(
                &mut module as *mut hipModule_t,
                image.as_ptr() as *const std::ffi::c_void
            )
        };
        error.into_result()?;
        Ok(Self { module })
    }
}

struct ContextFlags { 
    bits: u32 
}

impl ContextFlags {
    fn empty() -> Self {
        Self { bits: 0 }
    }
}

#[doc(hidden)]
pub struct Context {
    ctx: hipCtx_t            
}

impl Context {
    fn create_and_push(flags: ContextFlags, device: RocmDevice) -> RocmResult<Self> {
        let mut ctx: hipCtx_t = std::ptr::null_mut();
        let error = unsafe {
            hipCtxCreate(
                &mut ctx as *mut hipCtx_t,
                flags.bits,
                device.device
            )
        };
        match error {
            hipError_t::hipSuccess => Ok(Self { ctx }),
            _ => Err(error.into())
        }    
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let error = unsafe {
            hipCtxDestroy(self.ctx)
        };
        assert_eq!(error, hipError_t::hipSuccess);
    }
}

struct CurrentContext;

impl CurrentContext {
    fn set_current(c: &Context) -> RocmResult<()> {
        let error = unsafe {
            hipCtxSetCurrent(c.ctx)
        };
        error.into_result()
    }
}

#[doc(hidden)]
pub struct Hipblas {
    handle: hipblasHandle_t                        
}

impl Hipblas {
    fn new() -> RocmResult<Self> {
        let mut handle: hipblasHandle_t = std::ptr::null_mut();
        let status = unsafe {
            hipblasCreate(
                &mut handle as *mut hipblasHandle_t
            )
        };
        status.into_result()?;
        Ok(Self { handle })
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
                stream.stream as miopen_sys::hipStream_t
            )
        };
        status.into_result()?;
        Ok(Self { handle })     
    }
}

/// Safe wrapper for several ROCm implementation handles
pub struct RocmGpu {
    index: usize,
    device: RocmDevice,
    stream: Mutex<Stream>,
    kernels: Mutex<Module>,
    context: Mutex<Context>,
    hipblas: Mutex<Hipblas>,
    miopen: Mutex<Miopen>,
}

impl RocmGpu {
    pub fn new(index: usize) -> Arc<Self> {    
        let device = RocmDevice::get_device(index as u32)
            .expect(&format!("RocmGpu unable to get device {}!", index));
        let context = Context::create_and_push(ContextFlags::empty(), device)
            .expect("Unable to create Rocm Context!");
        let stream = Stream::new(StreamFlags::DEFAULT, Some(0))
            .expect("Unable to create Rocm Stream!");
        let src = CString::new(include_str!("rocm/kernels.s")).unwrap();
        let kernels = Module::load_from_string(&src).unwrap();
        let hipblas = Hipblas::new()
            .expect("Unable to create Hipblas!");
        let miopen = Miopen::with_stream(&stream)
            .expect("Unable to create Miopen!");
        Arc::new(Self {
            index,
            device,
            stream: Mutex::new(stream),
            kernels: Mutex::new(kernels),
            hipblas: Mutex::new(hipblas),
            miopen: Mutex::new(miopen),
            context: Mutex::new(context)
        })
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
