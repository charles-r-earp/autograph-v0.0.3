use crate::{
    Conv2dArgs, DataMut, DataRef, Num, Pool2dArgs, Tensor, Tensor4, TensorBase, TensorView1,
    TensorView4, TensorViewMut1, TensorViewMut4, Transpose, Unsigned,
};
use std::{sync::{Arc, Mutex, LockResult, MutexGuard, PoisonError}, ffi::{CString, CStr, c_void}, borrow::Cow, any::TypeId, fmt::{self, Debug}};

use hip_sys::hiprt::{
    hipError_t,
    hipMemsetD8,
    hipMemsetD32
};
use hip_sys::hipblas::{
    hipblasStatus_t,
    hipblasHandle_t,
    hipblasCreate,
    hipblasDestroy,
    hipblasSetStream,
    hipblasSaxpy,
    hipblasSgemm,
    hipblasOperation_t,
};
use miopen_sys::{
    miopenStatus_t,
    miopenHandle_t,
    miopenCreateWithStream,
    miopenDestroy,
    miopenTensorDescriptor_t,
    miopenCreateTensorDescriptor,
    miopenDestroyTensorDescriptor,
    miopenSetTensorDescriptor,
    miopenSet4dTensorDescriptor,
    miopenDataType_t,
    miopenConvolutionDescriptor_t,
    miopenCreateConvolutionDescriptor,
    miopenDestroyConvolutionDescriptor,
    miopenInitConvolutionDescriptor,
    miopenConvolutionMode_t,
    miopenConvSolution_t,
    miopenFindConvolutionForwardAlgorithm,
    miopenConvAlgoPerf_t,
    miopenConvFwdAlgorithm_t,
    miopenConvBwdDataAlgorithm_t,
    miopenConvBwdWeightsAlgorithm_t,
    miopenConvolutionForward,
    miopenConvolutionForwardGetWorkSpaceSize,
    miopenConvolutionForwardBias,
    miopenFindConvolutionBackwardDataAlgorithm,
    miopenConvolutionBackwardDataGetWorkSpaceSize,
    miopenConvolutionBackwardData,
    miopenConvolutionBackwardWeightsGetWorkSpaceSize,
    miopenFindConvolutionBackwardWeightsAlgorithm,
    miopenConvolutionBackwardWeights,
    miopenConvolutionBackwardBias,
    miopenActivationDescriptor_t,
    miopenCreateActivationDescriptor,
    miopenDestroyActivationDescriptor,
    miopenSetActivationDescriptor,
    miopenActivationMode_t,
    miopenActivationForward,
    miopenActivationBackward,
    miopenPoolingDescriptor_t,
    miopenCreatePoolingDescriptor,
    miopenDestroyPoolingDescriptor,
    miopenSet2dPoolingDescriptor,
    miopenPoolingMode_t,
    miopenPoolingGetWorkSpaceSizeV2,
    miopenPoolingForward,
    miopenPoolingBackward,
    miopenFusionPlanDescriptor_t,
    miopenCreateFusionPlan,
    miopenDestroyFusionPlan,
    miopenFusionDirection_t
};

use ndarray::{Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix4, IxDyn};

mod error;
use error::{RocmError, RocmResult, IntoResult};

#[doc(hidden)]
#[macro_use]
pub mod rustacuda_like;
pub(crate) use rustacuda_like::{DeviceCopy, DeviceSlice};
use rustacuda_like::{init, HipFlags, RocmDevice, Context, ContextFlags, CurrentContext, Stream, StreamFlags, Module, DevicePointer, DeviceBuffer, CopyDestination, launch}; 

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
    fn hipblas(&self) -> &Hipblas {
        &self.hipblas
    }
    fn miopen(&self) -> &Miopen {
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
        init(HipFlags::empty())
            .expect("Failed to initialize HIP!"); 
        let device = RocmDevice::get_device(index as u32)
            .expect(&format!("RocmGpu unable to get device {}!", index));        
        let context = Context::create_and_push(ContextFlags::empty(), device)
            .expect("Unable to create Rocm Context!");
        let stream = Stream::new(StreamFlags::empty(), Some(0))
            .expect("Unable to create Rocm Stream!");
        let src = unsafe {
            CStr::from_bytes_with_nul_unchecked(include_bytes!("rocm/kernels.hsaco"))
        };
        let kernels = Module::load_from_string(src)
            .expect("Unable to load kernels!");
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

impl Debug for RocmGpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RocmGpu({})", self.index)
    }
}

pub struct RocmBuffer<T: Num> {
    data: DeviceBuffer<T>,
    gpu: Arc<RocmGpu>
}

impl<T: Num> RocmBuffer<T> {
    pub(super) unsafe fn uninitialized(gpu: &Arc<RocmGpu>, len: usize) -> Self {
        let _gpu = gpu.lock()
            .unwrap();
        let data = DeviceBuffer::uninitialized(len).unwrap();
        let gpu = gpu.clone();
        Self { data, gpu }
    }
    pub(super) fn fill(&mut self, elem: T) {
        fn fill_u8(gpu: &RocmGpu, y: *mut u8, elem: u8, len: u32) {
            let gpu = gpu.lock()
                .unwrap();
            let stream = gpu.stream();
            let module = gpu.kernels();
            let (nblocks, nthreads) = get_nblocks_nthreads(len);
            unsafe {
                launch!(module.fill_u8<<<nblocks, nthreads, 0, stream>>>(
                    DevicePointer::wrap(y),
                    elem,
                    len
                )).unwrap();
            }
        }
        
        fn fill_u32(gpu: &RocmGpu, y: *mut u32, elem: u32, len: u32) {
            let gpu = gpu.lock()
                .unwrap();
            let stream = gpu.stream();
            let module = gpu.kernels();
            let (nblocks, nthreads) = get_nblocks_nthreads(len);
            unsafe {
                launch!(module.fill_u32<<<nblocks, nthreads, 0, stream>>>(
                    DevicePointer::wrap(y),
                    elem,
                    len
                )).unwrap();
            }
        }
        
        if TypeId::of::<T>() == TypeId::of::<u8>() {
            unsafe {    
                fill_u8(
                    &self.gpu, 
                    self.data.as_mut_ptr() as *mut u8,
                    elem.to_u8().unwrap(),
                    self.data.len() as u32
                ) 
            }
        }
        else if TypeId::of::<T>() == TypeId::of::<f32>() {
            unsafe {    
                fill_u32(
                    &self.gpu, 
                    self.data.as_mut_ptr() as *mut u32,
                    std::mem::transmute(elem.to_f32().unwrap()),
                    self.data.len() as u32
                ) 
            }
        }
        else {
            unreachable!()
        }
    }
    pub(super) fn len(&self) -> usize {
        self.data.len()
    }
    pub(super) fn as_device_slice(&self) -> &DeviceSlice<T> {
        &self.data
    }
    pub(super) fn as_mut_device_slice(&mut self) -> &mut DeviceSlice<T> {
        &mut self.data
    }
    pub(super) fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    pub(super) fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
    pub(super) fn to_vec(&self) -> Vec<T> {
        self.gpu.lock()
            .unwrap();
        let mut vec = Vec::with_capacity(self.data.len());
        unsafe { vec.set_len(self.data.len()) };
        self.data.copy_to(&mut vec);
        vec
    }
    pub(super) fn copy_from_slice<'a>(&mut self, slice: impl Into<Cow<'a, [T]>>) {
        let slice = slice.into();
        self.gpu.lock()
            .unwrap();
        self.data.copy_from(slice.as_ref()).unwrap();
    }
}

impl<T: Num> Clone for RocmBuffer<T> {
    fn clone(&self) -> Self {
        self.gpu.lock()
            .unwrap();
        let mut output = unsafe { Self::uninitialized(&self.gpu, self.data.len()) };
        self.data.copy_to(&mut output.data);
        output
    }
}

fn get_nblocks_nthreads(len: u32) -> (u32, u32) {
    const WARP_SIZE: u32 = 64;
    let nblocks = match len % WARP_SIZE {
        0 => len / WARP_SIZE,
        _ => len / WARP_SIZE + 1
    };
    (nblocks, WARP_SIZE)
}

struct TensorDescriptor {
    tensor_descriptor: miopenTensorDescriptor_t,
}

impl TensorDescriptor {
    fn new<D: IntoDimension>(dim: D, strides: Option<D>, data_type: miopenDataType_t) -> Self {
        let mut tensor_descriptor = unsafe { std::ptr::null_mut() };
        let status = unsafe {
            miopenCreateTensorDescriptor(
                &mut tensor_descriptor as *mut miopenTensorDescriptor_t
            )
        };
        status.into_result()
            .unwrap();
            
        let dim = dim.into_dimension();
        
        if strides.is_none() && dim.ndim() <= 4 {
            let [n, c, h, w] = match dim.slice() {
                &[n, c, h, w] => [n as i32, c as i32, h as i32, w as i32],
                &[n, c, h] => [n as i32, c as i32, h as i32, 1],
                &[n, c] => [n as i32, c as i32, 1, 1],
                &[c] => [1, c as i32, 1, 1],
                &[] => [1, 1, 1, 1],
                _ => unreachable!()
            };
            unsafe {
                miopenSet4dTensorDescriptor(
                    tensor_descriptor,
                    data_type, 
                    n, c, h, w
                ).into_result()
                    .unwrap();
            }
        }
        else {
            panic!();
            let strides = strides.map_or(
                dim.default_strides(),
                |strides| {
                    let strides = strides.into_dimension();
                    assert_eq!(dim.ndim(), strides.ndim());
                    strides
                }
            );

            let ndim = dim.ndim();
            
            let mut _dim = [0i32; 6];
            let mut _strides = [0i32; 6];
            
            dim.slice()
                .into_iter()
                .zip(strides.slice())
                .zip(
                    _dim.iter_mut()
                        .zip(_strides.iter_mut())
                )
                .for_each(|((d, s), (_d, _s))| {
                    *_d = *d as i32;
                    *_s = *s as i32;
                });
                
            
            let status = unsafe {
                miopenSetTensorDescriptor(
                    tensor_descriptor,
                    data_type,
                    ndim as i32,
                    _dim.as_ptr() as *mut i32,
                    _strides.as_ptr() as *mut i32
                )
            };
            status.into_result()
                .unwrap();
        }
        Self { tensor_descriptor }
    }
    unsafe fn as_mut_ptr(&self) -> miopenTensorDescriptor_t {
        self.tensor_descriptor
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        let status = unsafe { miopenDestroyTensorDescriptor(self.tensor_descriptor) };
        status.into_result()
            .unwrap();
    }
}

trait ConvolutionArgs {
    unsafe fn init_convolution_descriptor(&self, descriptor: miopenConvolutionDescriptor_t, mode: miopenConvolutionMode_t) -> miopenStatus_t;
}

impl ConvolutionArgs for Conv2dArgs {
    unsafe fn init_convolution_descriptor(&self, descriptor: miopenConvolutionDescriptor_t, mode: miopenConvolutionMode_t) -> miopenStatus_t {
        miopenInitConvolutionDescriptor(
            descriptor,
            mode,
            self.padding[0] as i32,
            self.padding[1] as i32,
            self.strides[0] as i32,
            self.strides[1] as i32,
            1, // dilation unused
            1, //
        )
    }
}

struct ConvolutionDescriptor {
    convolution_descriptor: miopenConvolutionDescriptor_t,
}

impl ConvolutionDescriptor {
    fn new(args: &impl ConvolutionArgs, mode: miopenConvolutionMode_t) -> Self {
        let mut convolution_descriptor = std::ptr::null_mut();
        let status = unsafe {
            miopenCreateConvolutionDescriptor(
                &mut convolution_descriptor as *mut miopenConvolutionDescriptor_t,
            )
        };
        status.into_result()
            .unwrap();
        let status = unsafe {
            args.init_convolution_descriptor(
                convolution_descriptor,
                mode
            )
        };
        status.into_result()
            .unwrap();
        Self {
            convolution_descriptor
        }
    }
    unsafe fn as_mut_ptr(&self) -> miopenConvolutionDescriptor_t {
        self.convolution_descriptor
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        let status = unsafe { miopenDestroyConvolutionDescriptor(self.convolution_descriptor) };
        status.into_result()
            .unwrap();
    }
}

// TODO: Potentially upstream to miopen-sys
// this might be fixed with bindgen
trait ConvAlgoPerfExt {
    unsafe fn fwd_algo(&self) -> miopenConvFwdAlgorithm_t;
    unsafe fn bwd_data_algo(&self) -> miopenConvBwdDataAlgorithm_t;
    unsafe fn bwd_weights_algo(&self) -> miopenConvBwdWeightsAlgorithm_t;
}

impl ConvAlgoPerfExt for miopenConvAlgoPerf_t {
    unsafe fn fwd_algo(&self) -> miopenConvFwdAlgorithm_t {
        self.__bindgen_anon_1.fwd_algo
    }
    unsafe fn bwd_data_algo(&self) -> miopenConvBwdDataAlgorithm_t {
        self.__bindgen_anon_1.bwd_data_algo
    }
    unsafe fn bwd_weights_algo(&self) -> miopenConvBwdWeightsAlgorithm_t {
        self.__bindgen_anon_1.bwd_weights_algo
    }
}

struct ActivationDescriptor {
    activation_descriptor: miopenActivationDescriptor_t,
}

impl ActivationDescriptor {
    fn new(
        mode: miopenActivationMode_t,
        alpha: Option<f64>,
        beta: Option<f64>,
        gamma: Option<f64>
    ) -> Self {
        let mut activation_descriptor = std::ptr::null_mut();
        let status = unsafe {
            miopenCreateActivationDescriptor(
                &mut activation_descriptor as *mut miopenActivationDescriptor_t,
            )
        };
        status.into_result()
            .unwrap();
        let status = unsafe {
            miopenSetActivationDescriptor(
                activation_descriptor,
                mode,
                alpha.unwrap_or(0.),
                beta.unwrap_or(0.),
                gamma.unwrap_or(0.)
            )
        };
        status.into_result()
            .unwrap();
        Self {
            activation_descriptor,
        }
    }
    unsafe fn as_mut_ptr(&self) -> miopenActivationDescriptor_t {
        self.activation_descriptor
    }
}

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        let status = unsafe {
            miopenDestroyActivationDescriptor(
                self.activation_descriptor
            )
        };
        status.into_result()
            .unwrap();
    }
}

trait PoolingArgs {
    unsafe fn set_pooling_descriptor(&self, descriptor: miopenPoolingDescriptor_t, mode: miopenPoolingMode_t) -> miopenStatus_t;
}

impl PoolingArgs for Pool2dArgs {
    unsafe fn set_pooling_descriptor(&self, descriptor: miopenPoolingDescriptor_t, mode: miopenPoolingMode_t) -> miopenStatus_t {
        miopenSet2dPoolingDescriptor( 
            descriptor,
            mode,
            self.kernel[0] as i32,
            self.kernel[1] as i32,
            self.padding[0] as i32,
            self.padding[1] as i32,
            self.strides[0] as i32,
            self.strides[1] as i32
        )
    }
}

struct PoolingDescriptor {
    pooling_descriptor: miopenPoolingDescriptor_t,
}

impl PoolingDescriptor {
    fn new(
        args: &impl PoolingArgs,
        mode: miopenPoolingMode_t,
    ) -> Self {
        let mut pooling_descriptor = unsafe { std::ptr::null_mut() };
        let status = unsafe {
            miopenCreatePoolingDescriptor(
                &mut pooling_descriptor as *mut miopenPoolingDescriptor_t
            )
        };
        status.into_result()
            .unwrap();
        let status = unsafe {
            args.set_pooling_descriptor(
                pooling_descriptor,
                mode
            )
        };
        status.into_result()
            .unwrap();
        Self { pooling_descriptor }
    }
    unsafe fn as_mut_ptr(&self) -> miopenPoolingDescriptor_t {
        self.pooling_descriptor
    }
}

impl Drop for PoolingDescriptor {
    fn drop(&mut self) {
        let status = unsafe { 
            miopenDestroyPoolingDescriptor(
                self.pooling_descriptor
            ) 
        };
        status.into_result()
            .unwrap();
    }
}

struct FusionPlanDescriptor {
    fusion_plan_descriptor: miopenFusionPlanDescriptor_t
}

impl FusionPlanDescriptor {
    fn new(direction: miopenFusionDirection_t, input_desc: &TensorDescriptor) -> Self {
        let mut fusion_plan_descriptor: miopenFusionPlanDescriptor_t = std::ptr::null_mut();
        unsafe {
            miopenCreateFusionPlan(
                &mut fusion_plan_descriptor as *mut miopenFusionPlanDescriptor_t,
                direction,
                input_desc.as_mut_ptr()
            ).into_result()
                .unwrap();
        }
        Self { fusion_plan_descriptor }
    } 
    unsafe fn as_mut_ptr(&self) -> miopenFusionPlanDescriptor_t {
        self.fusion_plan_descriptor
    }
}

impl Drop for FusionPlanDescriptor {
    fn drop(&mut self) {
        unsafe {
            miopenDestroyFusionPlan(
                self.fusion_plan_descriptor
            ).into_result()
                .unwrap()
        }
    }
}

pub(super) fn unsigned_to_f32<
    T: Unsigned,
    S1: DataRef<Elem = T>,
    S2: DataMut<Elem = f32>,
    D: Dimension,
>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D>,
) {
    let gpu = input.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let x = input.as_rocm_ptr().unwrap();
    let y = output.as_mut_rocm_ptr().unwrap();
    let len = input.len() as u32;
    let (nblocks, nthreads) = get_nblocks_nthreads(len);
    let stream = gpu.stream();
    let module = gpu.kernels();
    if TypeId::of::<T>() == TypeId::of::<u8>() {
        unsafe {
            launch!(module.u8_to_f32<<<nblocks, nthreads, 0, stream>>>(
              DevicePointer::wrap(x as *mut f32),
              DevicePointer::wrap(y),
              len
            ))
            .unwrap()
        }
    } else {
        unreachable!()
    }
}

pub(super) fn unsigned_to_one_hot_f32<
    T: Unsigned,
    S1: DataRef<Elem = T>,
    S2: DataMut<Elem = f32>,
>(
    input: &TensorBase<S1, Ix1>,
    output: &mut TensorBase<S2, Ix2>,
) {
    let (batch_size, nclasses) = output.dim();
    debug_assert_eq!(batch_size, input.dim());
    let gpu = input.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let x = input.as_rocm_ptr().unwrap();
    let y = output.as_mut_rocm_ptr().unwrap();
    let nclasses = nclasses as u32;
    let len = input.len() as u32;
    let (nblocks, nthreads) = get_nblocks_nthreads(len);
    let stream = gpu.stream();
    let module = gpu.kernels();
    if TypeId::of::<T>() == TypeId::of::<u8>() {
        unsafe {
            launch!(module.u8_to_one_hot_f32<<<nblocks, nthreads, 0, stream>>>(
              DevicePointer::wrap(x as *mut f32),
              nclasses,
              DevicePointer::wrap(y),
              len
            ))
            .unwrap()
        }
    } else {
        unreachable!()
    }
}

pub(super) fn broadcast<T: Num, D: Dimension, S1: DataRef<Elem = T>, S2: DataMut<Elem = T>>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D::Larger>,
) {
    let input = &input.as_rocm_slice().unwrap();
    output
        .as_mut_rocm_slice()
        .unwrap()
        .chunks_mut(input.len())
        .for_each(|mut output| {
            input.copy_to(output);
        });
}

pub(super) fn broadcast_backward<S1: DataMut<Elem = f32>, S2: DataRef<Elem = f32>, D: Dimension>(
    input_grad: &mut TensorBase<S1, D>,
    output_grad: &TensorBase<S2, D::Larger>,
) {
    let gpu = output_grad.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let alpha = unsafe { &1f32 as *const f32 };
    let dx = input_grad.as_mut_rocm_ptr().unwrap();
    let len = input_grad.len();
    output_grad
        .as_rocm_slice()
        .unwrap()
        .chunks(len)
        .for_each(|output_grad| unsafe {
            hipblasSaxpy(
                gpu.hipblas().as_mut_ptr(),
                len as i32,
                alpha,
                output_grad.as_ptr(),
                1,
                dx,
                1,
            );
        });
}

pub(super) fn gemm<S1: DataRef<Elem = f32>, S2: DataRef<Elem = f32>, S3: DataMut<Elem = f32>>(
    alpha: f32,
    a: &TensorBase<S1, Ix2>,
    trans_a: Transpose,
    b: &TensorBase<S2, Ix2>,
    trans_b: Transpose,
    beta: f32,
    c: &mut TensorBase<S3, Ix2>,
) {
    let (m, k1) = match trans_b {
        Transpose::Yes => b.dim(),
        Transpose::No => {
            let (k1, m) = b.dim();
            (m, k1)
        }
    };
    let ldb = match trans_b {
        Transpose::No => m,
        Transpose::Yes => k1,
    };
    let (k2, n) = match trans_a {
        Transpose::Yes => a.dim(),
        Transpose::No => {
            let (n, k2) = a.dim();
            (k2, n)
        }
    };
    let lda = match trans_a {
        Transpose::No => k2,
        Transpose::Yes => n,
    };
    debug_assert_eq!(k1, k2);
    debug_assert_eq!((n, m), c.dim());
    let gpu = a.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let hipblas = gpu.hipblas();
    let m = m as i32;
    let k = k1 as i32;
    let n = n as i32;
    let ldb = ldb as i32;
    let lda = lda as i32;
    let alpha = unsafe { &alpha as *const f32 };
    let beta = unsafe { &beta as *const f32 };
    let b = b.as_rocm_ptr().unwrap();
    let a = a.as_rocm_ptr().unwrap();
    let c = c.as_mut_rocm_ptr().unwrap();
    let trans_a = match trans_a {
        Transpose::Yes => hipblasOperation_t::HIPBLAS_OP_T,
        Transpose::No => hipblasOperation_t::HIPBLAS_OP_N,
    };
    let trans_b = match trans_b {
        Transpose::Yes => hipblasOperation_t::HIPBLAS_OP_T,
        Transpose::No => hipblasOperation_t::HIPBLAS_OP_N,
    };
    let status = unsafe {
        hipblasSgemm(
            hipblas.as_mut_ptr(),
            trans_b,
            trans_a,
            m,
            n,
            k,
            alpha,
            b,
            ldb,
            a,
            lda,
            beta,
            c,
            m,
        )
    };
    status.into_result()
            .unwrap();
}

// TODO Unit Test
pub(super) fn reduce_sum<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>, D: Dimension>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, Ix0>,
) {
    // may want to test using TensorOps for sums
    let gpu = input.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let mut len = input.len() / (2 * 256);
    if (input.len() % (2 * 256)) > 0 {
        len += 1;
    }
    let mut tmp = unsafe { DeviceBuffer::<f32>::uninitialized(len).unwrap() };
    let stream = gpu.stream();
    let module = gpu.kernels();
    {
        // partial sum
        let x = input.as_rocm_ptr().unwrap();
        let len = input.len() as u32;
        let nblocks = tmp.len() as u32;
        let nthreads = 256;
        unsafe {
            launch!(module.reduce_sum_partial<<<nblocks, nthreads, 0, stream>>>(
              DevicePointer::wrap(x as *mut f32),
              DevicePointer::wrap(tmp.as_mut_ptr()),
              len
            ))
            .unwrap()
        }
    }
    {
        // final sum
        let y = output.as_mut_rocm_ptr().unwrap();
        let len = len as u32;
        let nblocks = 1;
        let nthreads = 1;
        unsafe {
            launch!(module.reduce_sum_final<<<nblocks, nthreads, 0, stream>>>(
              DevicePointer::wrap(tmp.as_mut_ptr()),
              DevicePointer::wrap(y),
              len
            ))
            .unwrap()
        }
    }
}

pub(super) fn relu<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>, D: Dimension>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D>,
) {
    let gpu = input.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let miopen = gpu.miopen();
    let x_desc = TensorDescriptor::new(input.raw_dim(), None, miopenDataType_t::miopenFloat);
    let x = input.as_rocm_ptr().unwrap();
    let y_desc = TensorDescriptor::new(output.raw_dim(), None, miopenDataType_t::miopenFloat);
    let y = output.as_mut_rocm_ptr().unwrap();
    let relu_desc = ActivationDescriptor::new(
        miopenActivationMode_t::miopenActivationRELU,
        None,
        None,
        None
    );
    let status = unsafe {
        miopenActivationForward(
            miopen.as_mut_ptr(),
            relu_desc.as_mut_ptr(),
            &1f32 as *const f32 as *const _,
            x_desc.as_mut_ptr(),
            x as *const _,
            &0f32 as *const f32 as *const _,
            y_desc.as_mut_ptr(),
            y as *mut _,
        )
    };
    status.into_result()
        .unwrap();
}

pub(super) fn relu_backward<
    S1: DataRef<Elem = f32>,
    S2: DataMut<Elem = f32>,
    S3: DataRef<Elem = f32>,
    D: Dimension,
>(
    input: &TensorBase<S1, D>,
    input_grad: &mut TensorBase<S2, D>,
    output_grad: &TensorBase<S3, D>,
) {
    let gpu = input.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let miopen = gpu.miopen();
    let x_desc = TensorDescriptor::new(input.raw_dim(), None, miopenDataType_t::miopenFloat);
    let x = input.as_rocm_ptr().unwrap();
    let dx_desc = TensorDescriptor::new(input_grad.raw_dim(), None, miopenDataType_t::miopenFloat);
    let dx = input_grad.as_mut_rocm_ptr().unwrap();
    let dy_desc = TensorDescriptor::new(output_grad.raw_dim(), None, miopenDataType_t::miopenFloat);
    let dy = output_grad.as_rocm_ptr().unwrap();
    let relu_desc = ActivationDescriptor::new(
        miopenActivationMode_t::miopenActivationRELU,
        None,
        None,
        None
    );
    let status = unsafe {
        miopenActivationBackward(
            miopen.as_mut_ptr(),
            relu_desc.as_mut_ptr(),
            &1f32 as *const f32 as *const c_void,
            dy_desc.as_mut_ptr(),
            x as *const c_void,
            dy_desc.as_mut_ptr(),
            dy as *const c_void,
            x_desc.as_mut_ptr(),
            x as *const c_void,
            &0f32 as *const f32 as *const c_void,
            dx_desc.as_mut_ptr(),
            dx as *mut c_void,
        )
    };
    status.into_result()
        .unwrap();
}

pub(super) fn add<
    S1: DataRef<Elem = f32>,
    S2: DataRef<Elem = f32>,
    S3: DataMut<Elem = f32>,
    D: Dimension,
>(
    lhs: &TensorBase<S1, D>,
    rhs: &TensorBase<S2, D>,
    output: &mut TensorBase<S3, D>,
) {
    let gpu = lhs.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let x1 = lhs.as_rocm_ptr().unwrap();
    let x2 = rhs.as_rocm_ptr().unwrap();
    let y = output.as_mut_rocm_ptr().unwrap();
    let len = lhs.len() as u32;
    let (nblocks, nthreads) = get_nblocks_nthreads(len);
    let stream = gpu.stream();
    let module = gpu.kernels();
    unsafe {
        launch!(module.add<<<nblocks, nthreads, 0, stream>>>(
          DevicePointer::wrap(x1 as *mut f32),
          DevicePointer::wrap(x2 as *mut f32),
          DevicePointer::wrap(y),
          len
        ))
        .unwrap()
    }
}

pub(super) fn scaled_add<S1: DataMut<Elem = f32>, S2: DataRef<Elem = f32>, D: Dimension>(
    lhs: &mut TensorBase<S1, D>,
    alpha: f32,
    rhs: &TensorBase<S2, D>,
) {
    let gpu = rhs.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let hipblas = gpu.hipblas();
    let a = lhs.as_mut_rocm_ptr().unwrap();
    let alpha = unsafe { &alpha as *const f32 };
    let b = rhs.as_rocm_ptr().unwrap();
    let len = lhs.len() as i32;
    unsafe {
        hipblasSaxpy(hipblas.as_mut_ptr(), len, alpha, b, 1, a, 1);
    }
}

pub(super) fn cross_entropy<
    S1: DataRef<Elem = f32>,
    S2: DataRef<Elem = f32>,
    S3: DataMut<Elem = f32>,
>(
    input: &TensorBase<S1, Ix2>,
    target: &TensorBase<S2, Ix2>,
    output: &mut TensorBase<S3, Ix2>,
) {
    let gpu = input.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let stream = gpu.stream();
    let module = gpu.kernels();
    let (batch_size, nclasses) = input.dim();
    let (nblocks, nthreads) = get_nblocks_nthreads(batch_size as u32);
    let x = input.as_rocm_ptr().unwrap();
    let t = target.as_rocm_ptr().unwrap();
    let y = output.as_mut_rocm_ptr().unwrap();
    unsafe {
        launch!(module.cross_entropy_forward<<<nblocks, nthreads, 0, stream>>>(
          batch_size as u32,
          nclasses as u32,
          DevicePointer::wrap(x as *mut f32),
          DevicePointer::wrap(t as *mut f32),
          DevicePointer::wrap(y)
        ))
        .unwrap()
    }
}

pub(super) fn cross_entropy_backward<
    S1: DataRef<Elem = f32>,
    S2: DataMut<Elem = f32>,
    S3: DataRef<Elem = f32>,
    S4: DataRef<Elem = f32>,
>(
    input: &TensorBase<S1, Ix2>,
    input_grad: &mut TensorBase<S2, Ix2>,
    target: &TensorBase<S3, Ix2>,
    output_grad: &TensorBase<S4, Ix0>,
) {
    let gpu = input.device()
        .rocm()
        .unwrap()
        .lock()
        .unwrap();
    let stream = gpu.stream();
    let module = gpu.kernels();
    let len = input.len() as u32;
    let (batch_size, nclasses) = input.dim();
    let (nblocks, nthreads) = get_nblocks_nthreads(len);
    let x = input.as_rocm_ptr().unwrap();
    let dx = input_grad.as_mut_rocm_ptr().unwrap();
    let t = target.as_rocm_ptr().unwrap();
    let dy = output_grad.as_rocm_ptr().unwrap();
    unsafe {
        launch!(module.cross_entropy_backward<<<nblocks, nthreads, 0, stream>>>(
          DevicePointer::wrap(x as *mut f32),
          DevicePointer::wrap(dx),
          DevicePointer::wrap(t as *mut f32),
          DevicePointer::wrap(dy as *mut f32),
          len
        ))
        .unwrap()
    }
}

/*
fn reverse_conv2d_filter(input: &TensorView4<f32>, beta: f32, output: &mut TensorViewMut4<f32>) {
    let gpu = input.device()
        .rocm()
        .unwrap()
        .lock()
        .unwrap();
    let stream = gpu.stream();
    let module = gpu.kernels();
    let (outputs, inputs, kh, kw) = input.dim();
    let len = (outputs * inputs) as u32;
    let filter_len = (kh * kw) as u32;
    let (nblocks, nthreads) = get_nblocks_nthreads(len);
    let x = input.as_rocm_ptr().unwrap();
    let y = output.as_mut_rocm_ptr().unwrap();
    unsafe {
        launch!(module.reverse_conv_filter<<<nblocks, nthreads, 0, stream>>>(
          DevicePointer::wrap(x as *mut f32),
          beta,
          DevicePointer::wrap(y),
          filter_len,
          len
        ))
        .unwrap()
    }
}*/

const EXHAUSTIVE: bool = cfg!(feature = "exhaustive");

pub(super) fn conv2d<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    weight: &TensorView4<f32>,
    bias: Option<&TensorView1<f32>>,
    args: &Conv2dArgs,
    output: &mut TensorBase<S2, Ix4>,
) {
    let gpu = input.device.rocm()
        .unwrap()
        .lock()
        .unwrap();
    let miopen = gpu.miopen();
    let x_desc = TensorDescriptor::new(input.raw_dim(), None, miopenDataType_t::miopenFloat);
    let x = input.as_rocm_ptr().unwrap();
    let w_desc = TensorDescriptor::new(weight.raw_dim(), None, miopenDataType_t::miopenFloat);
    let w = weight.as_rocm_ptr().unwrap();
    let y_desc = TensorDescriptor::new(output.raw_dim(), None, miopenDataType_t::miopenFloat);
    let y = output.as_mut_rocm_ptr().unwrap();
    
    {
        let conv2d_desc = ConvolutionDescriptor::new(
            args, 
            miopenConvolutionMode_t::miopenConvolution
        );
        
        let mut workspace_size: usize = 0;
        
        unsafe {
            miopenConvolutionForwardGetWorkSpaceSize(
                miopen.as_mut_ptr(),
                w_desc.as_mut_ptr(),
                x_desc.as_mut_ptr(),
                conv2d_desc.as_mut_ptr(),
                y_desc.as_mut_ptr(),
                &mut workspace_size as *mut usize
            ).into_result()
                .unwrap();
        }
        
        let mut workspace = unsafe {
            DeviceBuffer::<u8>::uninitialized(workspace_size)
                .unwrap()
        };
        
        let mut perf: miopenConvAlgoPerf_t = unsafe { std::mem::uninitialized() };
        let requested_count = 1;
        let mut returned_count = 0i32;
        let exhaustive = EXHAUSTIVE;
        
        unsafe {
            miopenFindConvolutionForwardAlgorithm(
                miopen.as_mut_ptr(),
                x_desc.as_mut_ptr(),
                x as *const _,
                w_desc.as_mut_ptr(),
                w as *const _,
                conv2d_desc.as_mut_ptr(),
                y_desc.as_mut_ptr(),
                y as *mut _,
                requested_count,
                &mut returned_count as *mut _,
                &mut perf as *mut miopenConvAlgoPerf_t,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                exhaustive
            ).into_result()
                .unwrap();
        }
        assert_eq!(returned_count, 1);
        
        let algo = unsafe { perf.fwd_algo() };
        
        unsafe {
            miopenConvolutionForward(
                miopen.as_mut_ptr(),
                &1f32 as *const f32 as *const _,
                x_desc.as_mut_ptr(),
                x as *const _,
                w_desc.as_mut_ptr(),
                w as *const _,
                conv2d_desc.as_mut_ptr(),
                algo,
                &0f32 as *const f32 as *const _,
                y_desc.as_mut_ptr(),
                y as *mut _,
                workspace.as_mut_ptr() as *mut _,
                workspace_size
            ).into_result()
                .unwrap();
        }
    }    
    
    if let Some(bias) = bias {
        let b_desc = TensorDescriptor::new(
            bias.raw_dim(),
            None, 
            miopenDataType_t::miopenFloat
        ); 
        let b = bias.as_rocm_ptr().unwrap();
        
        let status = unsafe {
            miopenConvolutionForwardBias(
                miopen.as_mut_ptr(),
                &1f32 as *const f32 as *const _,
                b_desc.as_mut_ptr(),
                b as *const _,
                &1f32 as *const f32 as *const _,
                y_desc.as_mut_ptr(),
                y as *mut _
            )
        };
        status.into_result()
            .unwrap();
    }
}

pub(super) fn conv2d_backward_input<S1: DataMut<Elem = f32>>(
    input_grad: &mut TensorBase<S1, Ix4>,
    weight: &TensorView4<f32>,
    args: &Conv2dArgs,
    output_grad: &TensorView4<f32>,
) {
    let mut input_grad_tmp = unsafe { Tensor::uninitialized(input_grad.device(), input_grad.raw_dim()) };
    {
        let gpu = weight.device()
            .rocm()
            .unwrap()
            .lock()
            .unwrap();
        let miopen = gpu.miopen();
        let dx_desc = TensorDescriptor::new(input_grad.raw_dim(), None, miopenDataType_t::miopenFloat);
        let dx = input_grad_tmp.as_mut_rocm_ptr().unwrap();
        let w_desc = TensorDescriptor::new(weight.raw_dim(), None, miopenDataType_t::miopenFloat);
        let w = weight.as_rocm_ptr().unwrap();
        let dy_desc = TensorDescriptor::new(output_grad.raw_dim(), None, miopenDataType_t::miopenFloat);
        let dy = output_grad.as_rocm_ptr().unwrap();
        let mut conv2d_desc = ConvolutionDescriptor::new(args, miopenConvolutionMode_t::miopenConvolution);
        
        let mut workspace_size: usize = 0;
        
        unsafe {
            miopenConvolutionBackwardDataGetWorkSpaceSize(
                miopen.as_mut_ptr(),
                dy_desc.as_mut_ptr(),
                w_desc.as_mut_ptr(),
                conv2d_desc.as_mut_ptr(),
                dx_desc.as_mut_ptr(),
                &mut workspace_size as *mut usize
            ).into_result()
                .unwrap();
        }
        
        let mut workspace = unsafe {
            DeviceBuffer::<u8>::uninitialized(workspace_size)
                .unwrap()
        };
        
        let mut perf: miopenConvAlgoPerf_t = unsafe { std::mem::uninitialized() };
        let requested_count = 1;
        let mut returned_count = 0i32;
        let exhaustive = EXHAUSTIVE;
        
        unsafe {
            miopenFindConvolutionBackwardDataAlgorithm(
                miopen.as_mut_ptr(),
                dy_desc.as_mut_ptr(),
                dy as *const _,
                w_desc.as_mut_ptr(),
                w as *const _,
                conv2d_desc.as_mut_ptr(),
                dx_desc.as_mut_ptr(),
                dx as *mut _,
                requested_count,
                &mut returned_count as *mut _,
                &mut perf as *mut miopenConvAlgoPerf_t,
                workspace.as_mut_ptr() as *mut _,
                workspace_size,
                exhaustive
            ).into_result()
                .unwrap();
        }
        assert_eq!(returned_count, 1);
            
        let algo = unsafe { perf.bwd_data_algo() };
        
        unsafe {
            miopenConvolutionBackwardData(
                miopen.as_mut_ptr(),
                &1f32 as *const f32 as *const _,
                dy_desc.as_mut_ptr(),
                dy as *const _,
                w_desc.as_mut_ptr(),
                w as *const _,
                conv2d_desc.as_mut_ptr(),
                algo,
                &0f32 as *const f32 as *const _,
                dx_desc.as_mut_ptr(),
                dx as *mut _,
                workspace.as_mut_ptr() as *mut _,
                workspace_size
            ).into_result()
                .unwrap();
        }
    }
    input_grad.scaled_add(1., &input_grad_tmp.view());
}

pub(super) fn conv2d_backward_weight_bias<S1: DataRef<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    weight_grad: &mut TensorViewMut4<f32>,
    mut bias_grad: Option<&mut TensorViewMut1<f32>>,
    args: &Conv2dArgs,
    output_grad: &TensorView4<f32>,
) {
    let mut weight_grad_tmp = unsafe { Tensor::uninitialized(weight_grad.device(), weight_grad.raw_dim()) };
    let mut bias_grad_tmp = bias_grad.as_mut()
        .map(|bias_grad| unsafe { 
            Tensor::uninitialized(bias_grad.device(), bias_grad.raw_dim())
         });
    {
        let gpu = input.device()
            .rocm()
            .unwrap()
            .lock()
            .unwrap();
        let miopen = gpu.miopen();
        let x_desc = TensorDescriptor::new(input.raw_dim(), None, miopenDataType_t::miopenFloat);
        let x = input.as_rocm_ptr().unwrap();
        let dw_desc = TensorDescriptor::new(weight_grad.raw_dim(), None, miopenDataType_t::miopenFloat);
        let dw = weight_grad_tmp.as_mut_rocm_ptr().unwrap();
        let dy_desc = TensorDescriptor::new(output_grad.raw_dim(), None, miopenDataType_t::miopenFloat);
        let dy = output_grad.as_rocm_ptr().unwrap();
        
        {
            let conv2d_desc = ConvolutionDescriptor::new(args, miopenConvolutionMode_t::miopenConvolution);
        
            let mut workspace_size: usize = 0;
            
            unsafe {
                miopenConvolutionBackwardWeightsGetWorkSpaceSize(
                    miopen.as_mut_ptr(),
                    dy_desc.as_mut_ptr(),
                    x_desc.as_mut_ptr(),
                    conv2d_desc.as_mut_ptr(),
                    dw_desc.as_mut_ptr(),
                    &mut workspace_size as *mut usize
                ).into_result()
                    .unwrap();
            }
            
            let mut workspace = unsafe {
                DeviceBuffer::<u8>::uninitialized(workspace_size)
                    .unwrap()
            };
            
            let mut perf: miopenConvAlgoPerf_t = unsafe { std::mem::uninitialized() };
            let requested_count = 1;
            let mut returned_count = 0i32;
            let exhaustive = EXHAUSTIVE;
            
            unsafe {
                miopenFindConvolutionBackwardWeightsAlgorithm(
                    miopen.as_mut_ptr(),
                    dy_desc.as_mut_ptr(),
                    dy as *const _,
                    x_desc.as_mut_ptr(),
                    x as *const _,
                    conv2d_desc.as_mut_ptr(),
                    dw_desc.as_mut_ptr(),
                    dw as *mut _,
                    requested_count,
                    &mut returned_count as *mut _,
                    &mut perf as *mut miopenConvAlgoPerf_t,
                    workspace.as_mut_ptr() as *mut _,
                    workspace_size,
                    exhaustive
                ).into_result()
                    .unwrap();
            }
            assert_eq!(returned_count, 1);
                
            let algo = unsafe { perf.bwd_weights_algo() };
            
            unsafe {
                miopenConvolutionBackwardWeights(
                    miopen.as_mut_ptr(),
                    &1f32 as *const f32 as *const _,
                    dy_desc.as_mut_ptr(),
                    dy as *const _,
                    x_desc.as_mut_ptr(),
                    x as *const _,
                    conv2d_desc.as_mut_ptr(),
                    algo,
                    &0f32 as *const f32 as *const _,
                    dw_desc.as_mut_ptr(),
                    dw as *mut _,
                    workspace.as_mut_ptr() as *mut _,
                    workspace_size
                ).into_result()
                    .unwrap();
            }
        }
            
        if let Some(bias_grad) = &mut bias_grad_tmp {
            let db_desc = TensorDescriptor::new(
                bias_grad.raw_dim(),
                None,
                miopenDataType_t::miopenFloat
            );
            let db = bias_grad.as_mut_rocm_ptr().unwrap();
            
            let status = unsafe {
                miopenConvolutionBackwardBias(
                    miopen.as_mut_ptr(),
                    &1f32 as *const f32 as *const _,
                    dy_desc.as_mut_ptr(),
                    dy as *const _,
                    &0f32 as *const f32 as *const _,
                    db_desc.as_mut_ptr(),
                    db as *mut f32 as *mut _
                )
            };
            status.into_result()
                .unwrap();
        }
    }
    weight_grad.scaled_add(1., &weight_grad_tmp.view());
    match (bias_grad, bias_grad_tmp) {
        (Some(bias_grad), Some(bias_grad_tmp)) => {
            bias_grad.scaled_add(1., &bias_grad_tmp.view());
        },
        (None, None) => (),
        _ => unreachable!()
    }
}

pub(super) fn max_pool2d_forward<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    args: &Pool2dArgs,
    train: bool,
    output: &mut TensorBase<S2, Ix4>,
) -> Option<RocmBuffer<u8>> {
    let gpu = input.device()
        .rocm()
        .unwrap()
        .lock()
        .unwrap();
    let miopen = gpu.miopen();
    let x_desc = TensorDescriptor::new(input.raw_dim(), None, miopenDataType_t::miopenFloat);
    let x = input.as_rocm_ptr().unwrap();
    let y_desc = TensorDescriptor::new(output.raw_dim(), None, miopenDataType_t::miopenFloat);
    let y = output.as_mut_rocm_ptr().unwrap();
    let pool2d_desc = PoolingDescriptor::new(args, miopenPoolingMode_t::miopenPoolingMax);
    
    let mut workspace = if train {
        let mut workspace_size = 0usize;
         
        let status = unsafe {
            miopenPoolingGetWorkSpaceSizeV2(
                pool2d_desc.as_mut_ptr(),
                y_desc.as_mut_ptr(),
                &mut workspace_size as *mut usize
            )
        };
        status.into_result()
            .expect("Unable to get Pooling workspace size!");
        
        let workspace = unsafe {
            DeviceBuffer::<u8>::uninitialized(workspace_size)
                .expect("Unable to allocate PoolingForward workspace!")
        };
        Some(workspace)
    } else { None };
    
    let status = unsafe {
        miopenPoolingForward(
            miopen.as_mut_ptr(),
            pool2d_desc.as_mut_ptr(),
            &1f32 as *const f32 as *const _,
            x_desc.as_mut_ptr(),
            x as *const _,
            &0f32 as *const f32 as *const _,
            y_desc.as_mut_ptr(),
            y as *mut _,
            train,
            workspace.as_mut().map_or(std::ptr::null_mut(), |w| w.as_mut_ptr()) as *mut _,
            workspace.as_ref().map_or(0, |w| w.len())
        )
    };
    status.into_result()
        .unwrap();
    
    let gpu = input.device()
        .rocm()
        .unwrap()
        .clone();
    
    workspace.map(move |data| RocmBuffer { data, gpu })
}

pub(super) fn max_pool2d_backward<
    S1: DataRef<Elem = f32>,
    S2: DataMut<Elem = f32>,
    S3: DataRef<Elem = f32>,
>(
    input: &TensorBase<S1, Ix4>,
    input_grad: &mut TensorBase<S2, Ix4>,
    args: &Pool2dArgs,
    workspace: Option<&RocmBuffer<u8>>,
    output_grad: &TensorBase<S3, Ix4>,
) {
    let gpu = input.device()
        .rocm()
        .unwrap()
        .lock()
        .unwrap();
    let miopen = gpu.miopen();
    let x_desc = TensorDescriptor::new(input.raw_dim(), None, miopenDataType_t::miopenFloat);
    let x = input.as_rocm_ptr().unwrap();
    let dx_desc = TensorDescriptor::new(input_grad.raw_dim(), None, miopenDataType_t::miopenFloat);
    let dx = input_grad.as_mut_rocm_ptr().unwrap();
    let dy_desc = TensorDescriptor::new(output_grad.raw_dim(), None, miopenDataType_t::miopenFloat);
    let dy = output_grad.as_rocm_ptr().unwrap();
    let pool2d_desc = PoolingDescriptor::new(args, miopenPoolingMode_t::miopenPoolingMax);
    let status = unsafe {
        miopenPoolingBackward(
            miopen.as_mut_ptr(),
            pool2d_desc.as_mut_ptr(),
            &1f32 as *const f32 as *const c_void,
            dy_desc.as_mut_ptr(),
            x as *const c_void,
            dy_desc.as_mut_ptr(),
            dy as *const c_void,
            x_desc.as_mut_ptr(),
            x as *const c_void,
            &0f32 as *const f32 as *const c_void,
            dx_desc.as_mut_ptr(),
            dx as *mut c_void,
            workspace.map_or(std::ptr::null(), |w| w.as_ptr()) as *const _
        )
    };
    status.into_result()
        .unwrap();
}

pub(super) fn sgd_with_momentum<S1: DataMut<Elem=f32>, S2: DataRef<Elem=f32>, S3: DataMut<Elem=f32>, D: Dimension>
    (weight: &mut TensorBase<S1, D>, weight_grad: &TensorBase<S2, D>,
     learning_rate: f32, momentum: f32,
     velocity: &mut TensorBase<S3, D>) {
    let gpu = weight_grad.device()
        .rocm()
        .unwrap()
        .lock()
        .unwrap();
    let stream = gpu.stream();
    let module = gpu.kernels();
    let len = weight.len() as u32;
    let (nblocks, nthreads) = get_nblocks_nthreads(len);
    let mut w = weight.as_mut_rocm_ptr().unwrap();
    let dw = weight_grad.as_rocm_ptr().unwrap();
    let mut v = velocity.as_mut_rocm_ptr().unwrap();
    unsafe {
        launch!(module.sgd_with_momentum<<<nblocks, nthreads, 0, stream>>>(
          DevicePointer::wrap(w),
          DevicePointer::wrap(dw as *mut f32),
          learning_rate, 
          momentum,
          DevicePointer::wrap(v),
          len
        )).unwrap();
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
