use super::{
    Buffer, Conv2dArgs, DataMut, DataRef, Num, Pool2dArgs, Tensor2, TensorBase, TensorView1,
    Tensor, TensorView4, TensorViewMut1, TensorViewMut4, Transpose, Unsigned,
};
use ocl::{Platform, Device as OclDevice, Context, Queue, Buffer as OclBuffer, Program, Kernel};
use ocl::flags::MemFlags;
pub use ocl::flags::DeviceType as OclDeviceType;
use clblast_sys::{
    cl_mem, cl_command_queue, 
    CLBlastStatusCode::CLBlastSuccess, 
    CLBlastSgemm, CLBlastSsum, CLBlastSaxpy, CLBlastSaxpyBatched, CLBlastSconvgemm, 
    CLBlastLayout::*, CLBlastTranspose, CLBlastTranspose::*,
    CLBlastKernelMode::*
};
//use cpp::*;
use ndarray::{Dimension, Ix0, Ix1, Ix2, Ix4};
use num_traits::{Bounded, ToPrimitive};
use std::{
    borrow::Cow,
    fmt::{self, Debug},
    sync::{Arc, Mutex},
    any::TypeId
};

/*cpp!({
  #include <dnnl.hpp>
  #include <cassert>
  #include <utility>

  using dnnl_dt = dnnl::memory::data_type;
  using dnnl_tag = dnnl::memory::format_tag;
  using argmap = std::unordered_map<int, dnnl::memory>;
});*/

pub struct OpenclBuffer<T: Num> {
    data: OclBuffer<T>
}

impl<T: Num> Clone for OpenclBuffer<T> {
    fn clone(&self) -> Self {
        let data = OclBuffer::builder()
            .queue(self.data.default_queue().unwrap().clone())
            .len(self.data.len())
            .build()
            .unwrap();
        unsafe {
            self.data.copy(&data, None, None)
                .enq()
                .unwrap(); 
        }
        Self { data }
    }
}


impl<T: Num> OpenclBuffer<T> {
    pub(super) unsafe fn uninitialized(xpu: &OpenclXpu, len: usize) -> Self {
        let data = OclBuffer::builder()
            .queue(xpu.queue.clone())
            .len(len)
            .build()
            .unwrap();
        Self { data }
    }
    pub(super) fn from_vec<'a>(xpu: &OpenclXpu, slice: impl Into<Cow<'a, [T]>>) -> Self {
        let slice = slice.into();
        let data = OclBuffer::builder()
            .queue(xpu.queue.clone())
            .len(slice.len())
            .copy_host_slice(&slice)
            .build()
            .unwrap();
        Self { data }
    }
    pub(super) fn from_elem(xpu: &OpenclXpu, elem: T, len: usize) -> Self {
        let data = OclBuffer::builder()
            .queue(xpu.queue.clone())
            .fill_val(elem)
            .len(len)
            .build()
            .unwrap();
        Self { data }
    } 
    pub(super) fn len(&self) -> usize {
        self.data.len()
    }
    pub(super) fn fill(&mut self, elem: T) {
        unsafe {
            self.data.cmd()
                .fill(elem, None)
                .enq()
                .unwrap()
        }
    }
    pub(super) fn copy_from_slice<'a>(&mut self, slice: impl Into<Cow<'a, [T]>>) {
        let slice = slice.into();
        unsafe {
            self.data.write(slice.as_ref())
                .enq()
                .unwrap();
        }
    }
    pub(super) fn to_vec(&self) -> Vec<T> {
        let n = self.data.len();
        let mut vec = Vec::with_capacity(n);
        unsafe { vec.set_len(n) };
        unsafe {
            self.data.read(vec.as_mut_slice())
            .enq()
            .unwrap();
        }
        vec
    }
    pub(super) fn as_ocl_buffer(&self) -> OclBuffer<T> {
        let offset: usize = 0;
        let len = self.data.len();
        self.data.create_sub_buffer(
            Some(MemFlags::new().read_only().host_read_only()),
            offset,
            len
        ).unwrap()
    }
    pub(super) fn as_mut_ocl_buffer(&mut self) -> OclBuffer<T> {
        let offset: usize = 0;
        let len = self.data.len();
        self.data.create_sub_buffer(
            None,
            offset,
            len
        ).unwrap()
    }
}
/*
cpp_class!(pub unsafe struct OpenclEngine as "dnnl::engine");

impl OpenclEngine {
    fn new(context: &Context, device: OclDevice) -> Self {
        let context = unsafe { context.as_ptr() as onednn_sys::cl_context };
        let device_id = unsafe { device.as_raw() as onednn_sys::cl_device_id };
        cpp!(unsafe [context as "cl_context", device_id as "cl_device_id"] -> OpenclEngine as "dnnl::engine" {
          return dnnl::engine(dnnl::engine::kind::gpu, device_id, context);
        })
    }
}

cpp_class!(pub unsafe struct OpenclStream as "dnnl::stream");

impl OpenclStream {
    fn new(engine: &OpenclEngine, queue: &Queue) -> Self {
        let engine_ptr = unsafe { engine as *const OpenclEngine };
        let queue = unsafe { queue.as_ptr() as onednn_sys::cl_command_queue };
        cpp!(unsafe [engine_ptr as "const dnnl::engine*", queue as "cl_command_queue"] -> OpenclStream as "dnnl::stream" {
          auto engine = *engine_ptr;
          return dnnl::stream(engine, queue);
        })
    }
    fn wait(&mut self) {
        let stream_ptr = unsafe { self as *mut OpenclStream };
        cpp!(unsafe [stream_ptr as "dnnl::stream*"] {
            stream_ptr->wait();
        });
    }
}*/

/// 
pub struct OpenclXpu {
    //engine: OpenclEngine,
    //stream: Mutex<OpenclStream>,
    index: usize,
    queue: Queue,
    program: Program,
}

impl OpenclXpu {
    /// Constructs a new OpenclGpu wrapped in an Arc for threadsafe shared access
    pub fn new(index: usize, device_type: Option<OclDeviceType>) -> Arc<Self> {
        fn get_context(index: usize, device_type: Option<OclDeviceType>) -> Context {
            let mut i = 0;
            for platform in Platform::list() {
                let devices = OclDevice::list(&platform, device_type)
                    .unwrap();
                for device in devices {
                    if i == index {
                        let context = Context::builder()
                            .platform(platform)
                            .devices(device)
                            .build()
                            .unwrap();
                        return context;
                    }
                    else {
                        i += 1;
                    }
                }
            }
            panic!("OpenclXpu::new(): Invalid index!");
        }
        let context = get_context(index, device_type); 
        let device = context.devices()[0];
        let queue = Queue::new(&context, device, None).unwrap();
        //let engine = OpenclEngine::new(&context, device);
        //let stream = Mutex::new(OpenclStream::new(&engine, &queue));
        let program = Program::builder()
            .src(include_str!("opencl/kernels.cl"))
            .devices(device)
            .build(&context)
            .unwrap();
        Arc::new(Self { 
          //  engine, 
          //  stream,
            index,
            queue,
            program
        })
    }
    pub(super) fn synchronize(&self) {
        /*let mut stream = self.stream.lock()
            .unwrap();
        stream.wait();*/
        self.queue.finish()
            .unwrap(); 
    }
}

impl Debug for OpenclXpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let device = self.queue.device();
        //let vendor = device.vendor().unwrap();
        let name = device.name().unwrap();
        write!(f, "OpenclXpu({}, {})", self.index, name)
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
    let gpu = input.device.opencl().unwrap();
    let x = input.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    let len = input.len() as u32;
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    let name = if TypeId::of::<T>() == TypeId::of::<u8>() {
        "u8_to_f32"
    } 
    else {
        unreachable!()
    };
    let kernel = Kernel::builder()
        .program(&gpu.program)
        .name(name)
        .queue(gpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&x)
        .arg(&y)
        .arg(len)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
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
    let gpu = input.device.opencl().unwrap();
    let x = input.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    let nclasses = nclasses as u32;
    let len = input.len() as u32;
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    let name = if TypeId::of::<T>() == TypeId::of::<u8>() {
        "u8_to_one_hot_f32"
    } 
    else {
        unreachable!()
    };
    let kernel = Kernel::builder()
        .program(&gpu.program)
        .name(name)
        .queue(gpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&x)
        .arg(nclasses)
        .arg(&y)
        .arg(len)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
    }
}

pub(super) fn broadcast<T: Num, D: Dimension, S1: DataRef<Elem = T>, S2: DataMut<Elem = T>>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D::Larger>,
) {
    let batch_size = output.raw_dim()[0];
    let inputs = input.len();
    let x = input.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    (0 .. batch_size)
        .into_iter()
        .for_each(|b| {
            let dst_offset: usize = b * inputs;
            unsafe {
                x.copy(&y, Some(dst_offset), Some(inputs))
                    .enq()
                    .unwrap();
            }
        });
}

pub(super) fn broadcast_backward<S1: DataMut<Elem = f32>, S2: DataRef<Elem = f32>, D: Dimension>(
    input_grad: &mut TensorBase<S1, D>,
    output_grad: &TensorBase<S2, D::Larger>,
) {
    let xpu = output_grad.device.opencl().unwrap();
    let batch_size = output_grad.raw_dim()[0];
    
    let n = input_grad.len();
    let alpha = 1f32;
    let dy = output_grad.as_ocl_buffer().unwrap();
    let incdy = 1;
    let dx = input_grad.as_mut_ocl_buffer().unwrap();
    let incdx = 1;
    
    /*
    let nthreads = 64;
    let mut nblocks = n / nthreads;
    if n < nthreads || n % nthreads != 0 {
        nblocks += 1;
    }
    
    (0 .. batch_size)
        .into_iter()
        .for_each(|b| {
            let dy_offset: usize = b * n;
            /*let dy = dy.create_sub_buffer(
                Some(MemFlags::new().read_only()),
                dy_offset,
                n
            ).unwrap();
            let kernel = Kernel::builder()
                .program(&gpu.program)
                .name("axpy_f32")
                .queue(gpu.queue.clone())
                .global_work_size(nblocks * nthreads)
                .local_work_size(nthreads)
                .arg(n as u32)
                .arg(alpha)
                .arg(&dy)
                .arg(incdy as u32)
                .arg(&dx)
                .arg(incdx as u32)
                .build()
                .unwrap();
            unsafe {
                kernel.enq()
                    .unwrap();
            }*/
            let status = unsafe 
        });*/
    
    let offdx = 0;
    
    let mut command_queue = xpu.queue.as_ptr();
    let command_queue = unsafe { &mut command_queue as *mut *mut std::ffi::c_void };
    
    (0 .. batch_size).into_iter()
        .for_each(|b| {
            let offdy = b * n;
            let status = unsafe {
                CLBlastSaxpy(
                    n,
                    alpha,
                    dy.as_ptr() as cl_mem,
                    offdy,
                    incdy,
                    dx.as_ptr() as cl_mem,
                    offdx,
                    incdx,
                    command_queue as *mut cl_command_queue,
                    std::ptr::null_mut()
                )
            };
            
            assert_eq!(status, CLBlastSuccess);
            
            xpu.synchronize();
        });
}

impl Into<CLBlastTranspose> for Transpose {
    fn into(self) -> CLBlastTranspose {
        match self {
            Transpose::No => CLBlastTransposeNo,
            Transpose::Yes => CLBlastTransposeYes
        }
    }
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
    let (m, k1, lda) = match (trans_a, a.dim()) {
        (Transpose::No, (m, k)) => (m, k, k),
        (Transpose::Yes, (k, m)) => (m, k, m)
    };
    let (k2, n, ldb) = match (trans_b, b.dim()) {
        (Transpose::No, (k, n)) => (k, n, n),
        (Transpose::Yes, (n, k)) => (k, n, k)
    };
    let ldc = n;
    debug_assert_eq!(k1, k2);
    let k = k1;

    let xpu = a.device().opencl().unwrap();
    
    let a = a.as_ocl_buffer().unwrap();
    let b = b.as_ocl_buffer().unwrap();
    let c = c.as_mut_ocl_buffer().unwrap();
    
    let mut command_queue = xpu.queue.as_ptr();
    let command_queue = unsafe { &mut command_queue as *mut *mut std::ffi::c_void };
    
    let status = unsafe {
        CLBlastSgemm(
            CLBlastLayoutRowMajor,
            trans_a.into(),
            trans_b.into(),
            m, n, k,
            alpha,
            a.as_ptr() as cl_mem, 0, lda,
            b.as_ptr() as cl_mem, 0, ldb,
            beta,
            c.as_ptr() as cl_mem, 0, ldc,
            command_queue as *mut cl_command_queue,
            std::ptr::null_mut()
        )
    };
    assert_eq!(status, CLBlastSuccess);
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
    let xpu = input.device.opencl().unwrap();
    let (batch_size, nclasses) = input.dim();
    let nthreads = 64;
    let mut nblocks = batch_size / nthreads;
    if batch_size < nthreads || batch_size % nthreads != 0 {
        nblocks += 1;
    }
    let x = input.as_ocl_buffer().unwrap();
    let t = target.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    
    let kernel = Kernel::builder()
        .name("cross_entropy_forward")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(batch_size as u32)
        .arg(nclasses as u32)
        .arg(&x)
        .arg(&t)
        .arg(&y)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
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
    let xpu = input.device.opencl().unwrap();
    let len = input.len();
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    let x = input.as_ocl_buffer().unwrap();
    let dx = input_grad.as_mut_ocl_buffer().unwrap();
    let t = target.as_ocl_buffer().unwrap();
    let dy = output_grad.as_ocl_buffer().unwrap();
    
    let kernel = Kernel::builder()
        .name("cross_entropy_backward")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&x)
        .arg(&dx)
        .arg(&t)
        .arg(&dy)
        .arg(len as u32)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
    }
}

pub(super) fn reduce_sum<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>, D: Dimension>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, Ix0>,
) {
    let xpu = input.device.opencl().unwrap();
    let x = input.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    
    let mut command_queue = xpu.queue.as_ptr();
    let command_queue = unsafe { &mut command_queue as *mut *mut std::ffi::c_void };
    
    let status = unsafe {
        CLBlastSsum(
            x.len(),
            y.as_ptr() as cl_mem,
            0,
            x.as_ptr() as cl_mem,
            0,
            1,
            command_queue as *mut cl_command_queue,
            std::ptr::null_mut() 
        )
    };
    assert_eq!(status, CLBlastSuccess);
}

pub(super) fn relu<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>, D: Dimension>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D>,
) {
    let xpu = input.device.opencl().unwrap();
    let x = input.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    let len = x.len();
    
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    
    let kernel = Kernel::builder()
        .name("relu_forward")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .arg(&x)
        .arg(&y)
        .arg(len)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap()
    }
}

pub(super) fn scaled_add<S1: DataMut<Elem = f32>, S2: DataRef<Elem = f32>, D: Dimension>(
    lhs: &mut TensorBase<S1, D>,
    alpha: f32,
    rhs: &TensorBase<S2, D>,
) {
    let xpu = rhs.device.opencl().unwrap();
    let y = lhs.as_mut_ocl_buffer().unwrap();
    let x = rhs.as_ocl_buffer().unwrap();
    let n = lhs.len();
    
    let mut command_queue = xpu.queue.as_ptr();
    let command_queue = unsafe { &mut command_queue as *mut *mut std::ffi::c_void };
    
    let status = unsafe {
        CLBlastSaxpy(
            n,
            alpha,
            x.as_ptr() as cl_mem,
            0,
            1,
            y.as_ptr() as cl_mem,
            0,
            1,
            command_queue as *mut cl_command_queue,
            std::ptr::null_mut()
        )
    };
    assert_eq!(status, CLBlastSuccess);
}

pub(super) fn conv2d<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    weight: &TensorView4<f32>,
    bias: Option<&TensorView1<f32>>,
    args: &Conv2dArgs,
    output: &mut TensorBase<S2, Ix4>,
) {
    let xpu = input.device.opencl().unwrap();
    
    let (n, ic, ih, iw) = input.dim();
    let (_oc, _ic, kh, kw) = weight.dim();
    let (_n, oc, oh, ow) = output.dim();
    
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    let [dh, dw] = [1, 1];
    
    let weight = {
        // reverse kernel to get correct results
        let mut weight_reversed = unsafe { Tensor::<f32, _>::uninitialized(&weight.device, weight.raw_dim()) };
        let beta = 0f32;
        let filter_len = (kh * kw) as u32;
        let len = weight.len();
        let nthreads = 64;
        let mut nblocks = len / nthreads;
        if len < nthreads || len % nthreads != 0 {
            nblocks += 1;
        }
        let len = len as u32;
        
        let w = weight.as_ocl_buffer().unwrap();
        let w_rev = weight_reversed.as_mut_ocl_buffer().unwrap();
        
        let kernel = Kernel::builder()
            .name("reverse_conv_filter")
            .program(&xpu.program)
            .queue(xpu.queue.clone())
            .global_work_size(nblocks * nthreads)
            .local_work_size(nthreads)
            .arg(&w)
            .arg(0f32)
            .arg(&w_rev)
            .arg(filter_len)
            .arg(len)
            .build()
            .unwrap();
        unsafe {
            kernel.enq().unwrap();
        }
        weight_reversed
    };
    
    let x = input.as_ocl_buffer().unwrap();
    let w = weight.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    
    if let Some(bias) = &bias {
        let b = bias.as_ocl_buffer().unwrap();
        
        let nchannels = oc as u32;
        let filter_len = (kh * kw) as u32;
        let len = n * oc;
        let nthreads = 64;
        let mut nblocks = len / nthreads;
        if len < nthreads || len % nthreads != 0 {
            nblocks += 1;
        }
        let len = len as u32;
        
        let kernel = Kernel::builder()
            .name("conv2d_broadcast_bias")
            .program(&xpu.program)
            .queue(xpu.queue.clone())
            .global_work_size(nblocks * nthreads)
            .local_work_size(nthreads)
            .arg(&b)
            .arg(&y)
            .arg(nchannels)
            .arg(filter_len)
            .arg(len)
            .build()
            .unwrap();
        unsafe {
            kernel.enq().unwrap();
        }
    }
    
    let mut command_queue = xpu.queue.as_ptr();
    let command_queue = unsafe { &mut command_queue as *mut *mut std::ffi::c_void };
    
    let status = unsafe {
        CLBlastSconvgemm(
            CLBlastKernelModeConvolution,
            ic, ih, iw,
            kh, kw,
            ph, pw,
            sh, sw,
            dh, dw,
            oc, n,
            x.as_ptr() as cl_mem, 0,
            w.as_ptr() as cl_mem, 0,
            y.as_ptr() as cl_mem, 0,
            command_queue as *mut cl_command_queue,
            std::ptr::null_mut()
        )
    };
    
    assert_eq!(status, CLBlastSuccess);
}

pub(super) fn conv2d_backward_input<S1: DataMut<Elem = f32>>(
    input_grad: &mut TensorBase<S1, Ix4>,
    weight: &TensorView4<f32>,
    args: &Conv2dArgs,
    output_grad: &TensorView4<f32>,
) {
    let xpu = weight.device.opencl().unwrap();
    
    let (n, ic, ih, iw) = input_grad.dim();
    let (_oc, _ic, kh, kw) = weight.dim();
    let (_n, oc, oh, ow) = output_grad.dim();
    
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    let [dh, dw] = [1, 1];
    
    let weight = {
        // reverse kernel to get correct results
        let mut weight_reversed = unsafe { Tensor::<f32, _>::uninitialized(&weight.device, weight.raw_dim()) };
        let beta = 0f32;
        let filter_len = (kh * kw) as u32;
        let len = weight.len();
        let nthreads = 64;
        let mut nblocks = len / nthreads;
        if len < nthreads || len % nthreads != 0 {
            nblocks += 1;
        }
        let len = len as u32;
        
        let w = weight.as_ocl_buffer().unwrap();
        let w_rev = weight_reversed.as_mut_ocl_buffer().unwrap();
        
        let kernel = Kernel::builder()
            .name("reverse_conv_filter")
            .program(&xpu.program)
            .queue(xpu.queue.clone())
            .global_work_size(nblocks * nthreads)
            .local_work_size(nthreads)
            .arg(&w)
            .arg(0f32)
            .arg(&w_rev)
            .arg(filter_len)
            .arg(len)
            .build()
            .unwrap();
        unsafe {
            kernel.enq().unwrap();
        }
        weight_reversed
    };
    
    let dx = input_grad.as_mut_ocl_buffer().unwrap();
    let w = weight.as_ocl_buffer().unwrap();
    let dy = output_grad.as_ocl_buffer().unwrap();
    
    let mut command_queue = xpu.queue.as_ptr();
    let command_queue = unsafe { &mut command_queue as *mut *mut std::ffi::c_void };
    
    let status = unsafe {
        CLBlastSconvgemm(
            CLBlastKernelModeCrossCorrelation,
            ic, ih, iw,
            kh, kw,
            ph, pw,
            sh, sw,
            dh, dw,
            oc, n,
            dy.as_ptr() as cl_mem, 0,
            w.as_ptr() as cl_mem, 0,
            dx.as_ptr() as cl_mem, 0,
            command_queue as *mut cl_command_queue,
            std::ptr::null_mut()
        )
    };
    
    assert_eq!(status, CLBlastSuccess);
}
