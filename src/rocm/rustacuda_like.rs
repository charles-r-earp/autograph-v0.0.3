// Adapted from https://github.com/bheisler/RustaCUDA 
/*Copyright (c) 2018 Brook Heisler

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

use super::{RocmResult, IntoResult};

use hip_sys::hiprt::{
    hipError_t,
    hipMalloc,
    hipFree,
    hipDevice_t,
    hipDeviceGet,
    hipCtx_t,
    hipCtxCreate,
    hipCtxDestroy,
    hipCtxSetCurrent,
    hipStream_t,
    hipStreamCreateWithPriority,
    hipStreamDestroy,
    hipStreamSynchronize,
    hipModule_t,
    hipModuleLoadData,
    hipFunction_t,
    hipModuleGetFunction,
};
use std::{ffi::{CStr, c_void}, mem, slice::{Chunks, ChunksMut}, iter::{ExactSizeIterator, FusedIterator}, ops::{Deref, DerefMut}, marker::PhantomData};

#[derive(Clone, Copy)]
pub struct RocmDevice {
    device: hipDevice_t
}

impl RocmDevice {
    pub(super) fn get_device(ordinal: u32) -> RocmResult<Self> {
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

pub(super) struct StreamFlags {
    bits: u32
}

impl StreamFlags {
    pub(super) fn empty() -> Self { 
        Self { bits: 0 }
    }
}

pub struct Stream {
    stream: hipStream_t
}

impl Stream {
    pub(super) fn new(flags: StreamFlags, priority: Option<i32>) -> RocmResult<Self> {
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
    pub(super) fn synchronize(&self) -> RocmResult<()> {
        let error = unsafe {
            hipStreamSynchronize(self.stream)
        };
        error.into_result()
    }
    pub(super) unsafe fn as_mut_ptr(&self) -> hipStream_t {
        self.stream
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

pub struct Module {
    module: hipModule_t
}

impl Module {
    pub(super) fn load_from_string(image: &CStr) -> RocmResult<Self> {
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
    pub(super) fn get_function<'a>(&'a self, name: &CStr) -> RocmResult<Function<'a>> {
        let mut function: hipFunction_t = std::ptr::null_mut();
        let error = unsafe {
            hipModuleGetFunction(
                &mut function as *mut hipFunction_t,
                self.as_mut_ptr(),
                name.as_ptr() 
            )
        };
        error.into_result()?;
        Ok(Function { 
            function, 
            module: PhantomData::default() 
        })
    }
    unsafe fn as_mut_ptr(&self) -> hipModule_t {
        self.module
    }
}

pub(super) struct Function<'a> {
    function: hipFunction_t,
    module: PhantomData<&'a Module>
}

pub(super) struct ContextFlags { 
    bits: u32 
}

impl ContextFlags {
    pub(super) fn empty() -> Self {
        Self { bits: 0 }
    }
}

pub struct Context {
    ctx: hipCtx_t            
}

impl Context {
    pub(super) fn create_and_push(flags: ContextFlags, device: RocmDevice) -> RocmResult<Self> {
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

pub(super) struct CurrentContext;

impl CurrentContext {
    pub(super) fn set_current(c: &Context) -> RocmResult<()> {
        let error = unsafe {
            hipCtxSetCurrent(c.ctx)
        };
        error.into_result()
    }
}

pub unsafe trait DeviceCopy {}

#[repr(transparent)]
pub struct DevicePointer<T>(*mut T);

impl<T> DevicePointer<T> {
    pub(super) unsafe fn wrap(ptr: *mut T) -> Self {
        DevicePointer(ptr)
    }
    fn null() -> Self {
        DevicePointer(std::ptr::null_mut())
    }
    fn as_raw(&self) -> *const T {
        self.0
    }
    fn as_raw_mut(&mut self) -> *mut T {
        self.0
    }
    fn is_null(&self) -> bool {
        self.is_null()
    }
}

unsafe impl<T: DeviceCopy> DeviceCopy for DevicePointer<T> {}

pub struct DeviceSlice<T>([T]);

impl<T> DeviceSlice<T> {
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
    pub fn chunks(&self, chunk_size: usize) -> DeviceChunks<T> {
        DeviceChunks(self.0.chunks(chunk_size))
    }
    pub(super) fn chunks_mut(&mut self, chunk_size: usize) -> DeviceChunksMut<T> {
        DeviceChunksMut(self.0.chunks_mut(chunk_size))
    }
    unsafe fn from_slice(slice: &[T]) -> &DeviceSlice<T> {
        &*(slice as *const [T] as *const DeviceSlice<T>)
    }
    unsafe fn from_slice_mut(slice: &mut [T]) -> &mut DeviceSlice<T> {
        &mut *(slice as *mut [T] as *mut DeviceSlice<T>)
    }
}

#[derive(Debug, Clone)]
pub struct DeviceChunks<'a, T: 'a>(Chunks<'a, T>);
impl<'a, T> Iterator for DeviceChunks<'a, T> {
    type Item = &'a DeviceSlice<T>;

    fn next(&mut self) -> Option<&'a DeviceSlice<T>> {
        self.0
            .next()
            .map(|slice| unsafe { DeviceSlice::from_slice(slice) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize {
        self.0.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0
            .nth(n)
            .map(|slice| unsafe { DeviceSlice::from_slice(slice) })
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0
            .last()
            .map(|slice| unsafe { DeviceSlice::from_slice(slice) })
    }
}
impl<'a, T> DoubleEndedIterator for DeviceChunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a DeviceSlice<T>> {
        self.0
            .next_back()
            .map(|slice| unsafe { DeviceSlice::from_slice(slice) })
    }
}
impl<'a, T> ExactSizeIterator for DeviceChunks<'a, T> {}
impl<'a, T> FusedIterator for DeviceChunks<'a, T> {}

#[derive(Debug)]
pub struct DeviceChunksMut<'a, T: 'a>(ChunksMut<'a, T>);
impl<'a, T> Iterator for DeviceChunksMut<'a, T> {
    type Item = &'a mut DeviceSlice<T>;

    fn next(&mut self) -> Option<&'a mut DeviceSlice<T>> {
        self.0
            .next()
            .map(|slice| unsafe { DeviceSlice::from_slice_mut(slice) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize {
        self.0.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0
            .nth(n)
            .map(|slice| unsafe { DeviceSlice::from_slice_mut(slice) })
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0
            .last()
            .map(|slice| unsafe { DeviceSlice::from_slice_mut(slice) })
    }
}
impl<'a, T> DoubleEndedIterator for DeviceChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut DeviceSlice<T>> {
        self.0
            .next_back()
            .map(|slice| unsafe { DeviceSlice::from_slice_mut(slice) })
    }
}
impl<'a, T> ExactSizeIterator for DeviceChunksMut<'a, T> {}
impl<'a, T> FusedIterator for DeviceChunksMut<'a, T> {}

pub struct DeviceBuffer<T> {
    buf: DevicePointer<T>,
    capacity: usize,
}

impl<T> Deref for DeviceBuffer<T> {
    type Target = DeviceSlice<T>;

    fn deref(&self) -> &DeviceSlice<T> {
        unsafe {
            DeviceSlice::from_slice(::std::slice::from_raw_parts(
                self.buf.as_raw(),
                self.capacity,
            ))
        }
    }
}
impl<T> DerefMut for DeviceBuffer<T> {
    fn deref_mut(&mut self) -> &mut DeviceSlice<T> {
        unsafe {
            &mut *(::std::slice::from_raw_parts_mut(self.buf.as_raw_mut(), self.capacity)
                as *mut [T] as *mut DeviceSlice<T>)
        }
    }
}
impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.buf.is_null() {
            return;
        }

        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            // No choice but to panic if this fails.
            // This RustaCUDA impl seems unnecessary
            let mut ptr = mem::replace(&mut self.buf, DevicePointer::null());
            let error = unsafe {
                hipFree(ptr.as_raw_mut() as *mut c_void)
            };
            error.into_result()
                .expect("Failed to deallocate ROCm Device memory.");
        }
        self.capacity = 0;
    }
}

macro_rules! launch {
    ($module:ident . $function:ident <<<$grid:expr, $block:expr, $shared:expr, $stream:ident>>>( $( $arg:expr),* )) => {
        {
            let name = std::ffi::CString::new(stringify!($function)).unwrap();
            let function = $module.get_function(&name);
            match function {
                Ok(f) => launch!(f<<<$grid, $block, $shared, $stream>>>( $($arg),* ) ),
                Err(e) => Err(e),
            }
        }
    };
    ($function:ident <<<$grid:expr, $block:expr, $shared:expr, $stream:ident>>>( $( $arg:expr),* )) => {
        {
            //fn assert_impl_devicecopy<T: $crate::memory::DeviceCopy>(_val: T) {};
            fn assert_impl_devicecopy<T: $crate::rocm::DeviceCopy>(_val: T) {};
            if false {
                $(
                    assert_impl_devicecopy($arg);
                )*
            };

            $stream.launch(&$function, $grid, $block, $shared,
                &[
                    $(
                        &$arg as *const _ as *mut ::std::ffi::c_void,
                    )*
                ]
            )
        }
    };
}
