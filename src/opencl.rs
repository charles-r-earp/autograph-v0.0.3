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
    CLBlastSgemm, CLBlastSsum, CLBlastSaxpy, CLBlastSaxpyBatched, CLBlastSconvgemm, CLBlastSgemmBatched, CLBlastSim2col, CLBlastScol2im, 
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
    let xpu = input.device.opencl().unwrap();
    
    let c = input.len();
    let x = input.as_ocl_buffer().unwrap();
    let len = output.len();
    let y = output.as_mut_ocl_buffer().unwrap();
    
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    
    let kernel = Kernel::builder()
        .program(&xpu.program)
        .name("broadcast")
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&x)
        .arg(&y)
        .arg(c as u32)
        .arg(len as u32)
        .build()
        .unwrap();
    unsafe {
        kernel.enq()
            .unwrap();
    }
}

pub(super) fn broadcast_backward<S1: DataMut<Elem = f32>, S2: DataRef<Elem = f32>, D: Dimension>(
    input_grad: &mut TensorBase<S1, D>,
    output_grad: &TensorBase<S2, D::Larger>,
) {
    let xpu = output_grad.device.opencl().unwrap();
    
    let c = input_grad.len();
    let dx = input_grad.as_mut_ocl_buffer().unwrap();
    let len = output_grad.len();
    let dy = output_grad.as_ocl_buffer().unwrap();
    
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    
    let kernel = Kernel::builder()
        .program(&xpu.program)
        .name("broadcast_backward")
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&dx)
        .arg(&dy)
        .arg(c as u32)
        .arg(len as u32)
        .build()
        .unwrap();
    unsafe {
        kernel.enq()
            .unwrap();
    }
    /*
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
        });*/
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
        .arg(len as u32)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap()
    }
}

pub(super) fn relu_backward<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>, S3: DataRef<Elem = f32>, D: Dimension>(
    input: &TensorBase<S1, D>,
    input_grad: &mut TensorBase<S2, D>,
    output_grad: &TensorBase<S3, D>,
) {
    let xpu = input.device.opencl().unwrap();
    let x = input.as_ocl_buffer().unwrap();
    let dx = input_grad.as_mut_ocl_buffer().unwrap();
    let dy = output_grad.as_ocl_buffer().unwrap();
    let len = x.len();
    
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    
    let kernel = Kernel::builder()
        .name("relu_backward")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .arg(&x)
        .arg(&dx)
        .arg(&dy)
        .arg(len as u32)
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
    
    /*
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
    */
    // naive custom opencl saxpy is faster (at least for n small), clblast xaxpy optimizes for specific multiples and may not be as efficient with arbitrary n
    let len = n;
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    let incx = 1u32;
    let incy = 1u32;
    let offx = 0u32;
    let offy = 0u32;
    
    let kernel = Kernel::builder()
        .name("axpy_f32")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .arg(n as u32)
        .arg(alpha)
        .arg(&x)
        .arg(offx)
        .arg(incx)
        .arg(&y)
        .arg(offy)
        .arg(incy)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap()
    }
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
    
    /*let weight = {
        // reverse kernel to get correct results
        let mut weight_reversed = unsafe { Tensor::<f32, _>::uninitialized(&weight.device, weight.raw_dim()) };
        let beta = 0f32;
        let filter_len = kh * kw;
        let len = weight.len();
        let nthreads = 64;
        let mut nblocks = len / nthreads;
        if len < nthreads || len % nthreads != 0 {
            nblocks += 1;
        }
        
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
            .arg(filter_len as u32)
            .arg(len as u32)
            .build()
            .unwrap();
        unsafe {
            kernel.enq().unwrap();
        }
        weight_reversed
    };*/
    
    let x = input.as_ocl_buffer().unwrap();
    let w = weight.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    
    if let Some(bias) = &bias {
        let b = bias.as_ocl_buffer().unwrap();
        
        let nchannels = oc;
        let image_len = oh * ow;
        let len = n * oc;
        let nthreads = 64;
        let mut nblocks = len / nthreads;
        if len < nthreads || len % nthreads != 0 {
            nblocks += 1;
        }
        
        let kernel = Kernel::builder()
            .name("conv2d_broadcast_bias")
            .program(&xpu.program)
            .queue(xpu.queue.clone())
            .global_work_size(nblocks * nthreads)
            .local_work_size(nthreads)
            .arg(&b)
            .arg(&y)
            .arg(nchannels as u32)
            .arg(image_len as u32)
            .arg(len as u32)
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
            //CLBlastKernelModeConvolution,
            CLBlastKernelModeCrossCorrelation,
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

/*
fn nchw_to_nhwc<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>>(input: &TensorBase<S1, Ix4>, output: &mut TensorBase<S2, Ix4>) {
    let xpu = input.device().opencl().unwrap();
    let (n, c, h, w) = input.dim();
    debug_assert_eq!(output.dim(), (n, h, w, c));
    
    let len = n*c;
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    
    let x = input.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    
    let kernel = Kernel::builder()
        .name("nchw_to_nhwc")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&x)
        .arg(&y)
        .arg(n as u32)
        .arg(c as u32)
        .arg(h as u32)
        .arg(w as u32)
        .arg(len as u32)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
    }
}   

fn nhwc_to_nchw<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>>(input: &TensorBase<S1, Ix4>, beta: f32, output: &mut TensorBase<S2, Ix4>) {
    let xpu = input.device().opencl().unwrap();
    let (n, h, w, c) = input.dim();
    debug_assert_eq!(output.dim(), (n, c, h, w));
    
    let len = n*c;
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    
    let x = input.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    
    let kernel = Kernel::builder()
        .name("nhwc_to_nchw")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&x)
        .arg(&y)
        .arg(beta)
        .arg(n as u32)
        .arg(c as u32)
        .arg(h as u32)
        .arg(w as u32)
        .arg(len as u32)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
    }
}   

fn ohwi_flipped_to_oihw<S1: DataRef<Elem=f32>, S2: DataMut<Elem=f32>>(input: &TensorBase<S1, Ix4>, beta: f32, output: &mut TensorBase<S2, Ix4>) {
    let xpu = input.device().opencl().unwrap();
    let (oc, kh, kw, ic) = input.dim();
    debug_assert_eq!(output.dim(), (oc, ic, kh, kw));
    
    let len = oc*ic;
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    
    let x = input.as_ocl_buffer().unwrap();
    let y = output.as_mut_ocl_buffer().unwrap();
    
    let kernel = Kernel::builder()
        .name("ohwi_flipped_to_oihw")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&x)
        .arg(&y)
        .arg(beta)
        .arg(oc as u32)
        .arg(ic as u32)
        .arg(kh as u32)
        .arg(kw as u32)
        .arg(len as u32)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
    }
}*/  

pub(super) fn conv2d_backward_input<S1: DataMut<Elem = f32>>(
    input_grad: &mut TensorBase<S1, Ix4>,
    weight: &TensorView4<f32>,
    args: &Conv2dArgs,
    output_grad: &TensorView4<f32>,
) {
    //return;
    // im2col
    // input [n, ic, ih, iw]
    // input_col [n, ic, oh, ow, kh, kw] => [n*ic*oh*ow, kh*kw] 
    // weight [oc, ic, kh, kw] => [oc*ic, kh*kw]
    // [n, ic*kh*kw, oh*ow] [ic*kh*kw, oc] => [n, oh*ow, oc]
    // [oc, ic*kh*kw] [n, ic*kh*kw, oh*ow] => [n, oc, oh*ow] 
    let device = weight.device();
    let xpu = device.opencl().unwrap();
    
    let (n, ic, ih, iw) = input_grad.dim();
    let (_oc, _ic, kh, kw) = weight.dim();
    let (_n, oc, oh, ow) = output_grad.dim();
    
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    
    let dx = input_grad.as_mut_ocl_buffer().unwrap();
    let w = weight.as_ocl_buffer().unwrap();
    let dy = output_grad.as_ocl_buffer().unwrap();
    
    if true { // uses more memory but faster
        let mut dx_col = OclBuffer::<f32>::builder()
            .queue(xpu.queue.clone())
            .len(n * ic*kh*kw * oh*ow)
            .build()
            .unwrap();

        let mut command_queue = xpu.queue.as_ptr();
        let command_queue = unsafe { &mut command_queue as *mut *mut std::ffi::c_void };
        
        { // dx_col = gemm wT * dy[n]
            let batch_size = n;
            let alphas = vec![1.; batch_size];
            let betas = vec![0.; batch_size];
            let dy_offsets: Vec<usize> = (0 .. batch_size)
                .into_iter()
                .map(|batch| batch*oc*oh*ow)
                .collect();
            let w_offsets = vec![0; batch_size];
            let dx_col_offsets: Vec<usize> = (0 .. batch_size)
                .into_iter()
                .map(|batch| batch*ic*kh*kw*oh*ow)
                .collect();
            let m = ic*kh*kw;
            let k = oc;
            let n = oh*ow;
            let lda = m;
            let ldb = n;
            let ldc = n;
            
            let status = unsafe {
                CLBlastSgemmBatched(
                    CLBlastLayoutRowMajor,
                    Transpose::Yes.into(),
                    Transpose::No.into(),
                    m, n, k,
                    alphas.as_ptr(),
                    w.as_ptr() as cl_mem, w_offsets.as_ptr(), lda,
                    dy.as_ptr() as cl_mem, dy_offsets.as_ptr(), ldb,
                    betas.as_ptr(),
                    dx_col.as_ptr() as cl_mem, dx_col_offsets.as_ptr(), ldc,
                    batch_size,
                    command_queue as *mut cl_command_queue,
                    std::ptr::null_mut()
                )
            };
            assert_eq!(status, CLBlastSuccess);
        }
        
        (0 .. n).into_iter()
            .for_each(|batch| {
                // dx += dx_col.col2im()
                let dx_col_offset = batch*ic*kh*kw*oh*ow;
                let dx_offset = batch*ic*ih*iw;
                let status = unsafe {
                    CLBlastScol2im(
                        CLBlastKernelModeCrossCorrelation,
                        ic, ih, iw,
                        kh, kw,
                        ph, pw,
                        sh, sw,
                        1, 1, // dilation unused
                        dx_col.as_ptr() as cl_mem, dx_col_offset,
                        dx.as_ptr() as cl_mem, dx_offset,
                        command_queue as *mut cl_command_queue,
                        std::ptr::null_mut()
                    )
                };
                assert_eq!(status, CLBlastSuccess); 
            });
    }
    else {
    
        let mut dx_col = OclBuffer::<f32>::builder()
            .queue(xpu.queue.clone())
            .len(ic*kh*kw * oh*ow)
            .build()
            .unwrap();
        let mut dx_tmp = OclBuffer::<f32>::builder()
            .queue(xpu.queue.clone())
            .len(n * ic * ih * iw)
            .build()
            .unwrap();
        
        
        let mut command_queue = xpu.queue.as_ptr();
        let command_queue = unsafe { &mut command_queue as *mut *mut std::ffi::c_void };
        
        (0 .. n).into_iter()
            .for_each(|batch| {
                { // dx_col = gemm wT * dy[n]
                    let alpha = 1.;
                    let beta = 0.;
                    let dy_offset = batch*oc*oh*ow;
                    let m = ic*kh*kw;
                    let k = oc;
                    let n = oh*ow;
                    let lda = m;
                    let ldb = n;
                    let ldc = n;
                    
                    let status = unsafe {
                        CLBlastSgemm(
                            CLBlastLayoutRowMajor,
                            Transpose::Yes.into(),
                            Transpose::No.into(),
                            m, n, k,
                            alpha,
                            w.as_ptr() as cl_mem, 0, lda,
                            dy.as_ptr() as cl_mem, dy_offset, ldb,
                            beta,
                            dx_col.as_ptr() as cl_mem, 0, ldc,
                            command_queue as *mut cl_command_queue,
                            std::ptr::null_mut()
                        )
                    };
                    assert_eq!(status, CLBlastSuccess);
                }
                
                { // dx += dx_col.col2im()
                    let dx_offset = batch*ic*ih*iw;
                    let status = unsafe {
                        CLBlastScol2im(
                            CLBlastKernelModeCrossCorrelation,
                            ic, ih, iw,
                            kh, kw,
                            ph, pw,
                            sh, sw,
                            1, 1, // dilation unused
                            dx_col.as_ptr() as cl_mem, 0,
                            dx.as_ptr() as cl_mem, dx_offset,
                            command_queue as *mut cl_command_queue,
                            std::ptr::null_mut()
                        )
                    };
                    assert_eq!(status, CLBlastSuccess); 
                }
                
            });
    }
    
    
}

pub(super) fn conv2d_backward_weight_bias<S1: DataRef<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    weight_grad: &mut TensorViewMut4<f32>,
    bias_grad: Option<&mut TensorViewMut1<f32>>,
    args: &Conv2dArgs,
    output_grad: &TensorView4<f32>,
) {
    // [n, oc, oh*ow] [n, (ic*kh*kw, oh*ow)T] => [oc, ic*kh*kw]
    
    let device = input.device();
    let xpu = device.opencl().unwrap();
    
    let (n, ic, ih, iw) = input.dim();
    let (_oc, _ic, kh, kw) = weight_grad.dim();
    let (_n, oc, oh, ow) = output_grad.dim();
    
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    
    let mut command_queue = xpu.queue.as_ptr();
    let command_queue = unsafe { &mut command_queue as *mut *mut std::ffi::c_void };
    
    let x = input.as_ocl_buffer().unwrap();
    let mut dw = weight_grad.as_mut_ocl_buffer().unwrap();
    let dy = output_grad.as_ocl_buffer().unwrap();
    
    /*
    let mut x_col = OclBuffer::<f32>::builder()
        .queue(xpu.queue.clone())
        .len(ic*kh*kw * oh*ow)
        .build()
        .unwrap();
    
    (0 .. n).into_iter()
        .for_each(|batch| {
        { // x_col = x[n].im2col
            let x_offset = batch*ic*ih*iw;
            let status = unsafe {
                CLBlastSim2col(
                    CLBlastKernelModeCrossCorrelation,
                    ic, ih, iw,
                    kh, kw,
                    ph, pw,
                    sh, sw,
                    1, 1, // dilation unused
                    x.as_ptr() as cl_mem, x_offset,
                    x_col.as_ptr() as cl_mem, 0,
                    command_queue as *mut cl_command_queue,
                    std::ptr::null_mut()
                )
            };
            assert_eq!(status, CLBlastSuccess);
        }
        { // dw += gemm dy[n] * x_colT
                let alpha = 1.;
                let beta = 1.;
                let dy_offset = batch*oc*oh*ow;
                let m = oc;
                let k = oh*ow;
                let n = ic*kh*kw;
                let lda = k;
                let ldb = k;
                let ldc = n;
                
                let status = unsafe {
                    CLBlastSgemm(
                        CLBlastLayoutRowMajor,
                        Transpose::No.into(),
                        Transpose::Yes.into(),
                        m, n, k,
                        alpha,
                        dy.as_ptr() as cl_mem, dy_offset, lda,
                        x_col.as_ptr() as cl_mem, 0, ldb,
                        beta,
                        dw.as_ptr() as cl_mem, 0, ldc,
                        command_queue as *mut cl_command_queue,
                        std::ptr::null_mut()
                    )
                };
                assert_eq!(status, CLBlastSuccess);
            }
    }); 
    */
    
    let mut x_col = OclBuffer::<f32>::builder()
        .queue(xpu.queue.clone())
        .len(n*ic*kh*kw * oh*ow)
        .build()
        .unwrap();
    
    (0 .. n).into_iter()
        .for_each(|batch| {
        { // x_col = x[n].im2col
            let x_offset = batch*ic*ih*iw;
            let x_col_offset = batch*ic*kh*kw*oh*ow;
            let status = unsafe {
                CLBlastSim2col(
                    CLBlastKernelModeCrossCorrelation,
                    ic, ih, iw,
                    kh, kw,
                    ph, pw,
                    sh, sw,
                    1, 1, // dilation unused
                    x.as_ptr() as cl_mem, x_offset,
                    x_col.as_ptr() as cl_mem, x_col_offset,
                    command_queue as *mut cl_command_queue,
                    std::ptr::null_mut()
                )
            };
            assert_eq!(status, CLBlastSuccess);
        }
        
    }); 
    
    { // dw += gemm dy[n] * x_col[n]T
        let batch_size = n;
        let alphas = vec![1.; batch_size];
        let betas = vec![1.; batch_size];
        let dy_offsets: Vec<usize> = (0 .. batch_size).into_iter()
            .map(|batch| batch*oc*oh*ow)
            .collect();
        let x_col_offsets: Vec<usize> = (0 .. batch_size).into_iter()
            .map(|batch| batch*oc*oh*ow)
            .collect();
        let dw_offsets = vec![0; batch_size];
        let m = oc;
        let k = oh*ow;
        let n = ic*kh*kw;
        let lda = k;
        let ldb = k;
        let ldc = n;
        
        let status = unsafe {
            CLBlastSgemmBatched(
                CLBlastLayoutRowMajor,
                Transpose::No.into(),
                Transpose::Yes.into(),
                m, n, k,
                alphas.as_ptr(),
                dy.as_ptr() as cl_mem, dy_offsets.as_ptr(), lda,
                x_col.as_ptr() as cl_mem, x_col_offsets.as_ptr(), ldb,
                betas.as_ptr(),
                dw.as_ptr() as cl_mem, dw_offsets.as_ptr(), ldc,
                batch_size,
                command_queue as *mut cl_command_queue,
                std::ptr::null_mut()
            )
        };
        assert_eq!(status, CLBlastSuccess);
    }
    
    
    if let Some(bias_grad) = bias_grad {
        let mut db = bias_grad.as_mut_ocl_buffer().unwrap();
        
        let nchannels = oc;
        let image_len = oh * ow;
        let len = n * oc;
        let nthreads = 64;
        let mut nblocks = len / nthreads;
        if len < nthreads || len % nthreads != 0 {
            nblocks += 1;
        }
        
        let kernel = Kernel::builder()
            .name("conv2d_broadcast_bias_backward")
            .program(&xpu.program)
            .queue(xpu.queue.clone())
            .global_work_size(nblocks * nthreads)
            .local_work_size(nthreads)
            .arg(&db)
            .arg(&dy)
            .arg(nchannels as u32)
            .arg(image_len as u32)
            .arg(len as u32)
            .build()
            .unwrap();
        unsafe {
            kernel.enq().unwrap();
        }
    }
}

pub(super) fn max_pool2d<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    args: &Pool2dArgs,
    output: &mut TensorBase<S2, Ix4>,
) {
    let xpu = input.device.opencl().unwrap();
    
    let x = input.as_ocl_buffer().unwrap();
    let y = output.as_ocl_buffer().unwrap();
    
    let (n, ic, ih, iw) = input.dim();
    let (_n, oc, oh, ow) = output.dim();
    
    let [kh, kw] = args.kernel;
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    
    let len = n * oc;
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    
    let kernel = Kernel::builder()
        .name("max_pool2d_forward")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&x)
        .arg(&y)
        .arg(ih as u32)
        .arg(iw as u32)
        .arg(oh as u32)
        .arg(ow as u32)
        .arg(kh as u32)
        .arg(kw as u32)
        .arg(sh as u32)
        .arg(sw as u32)
        .arg(ph as u32)
        .arg(pw as u32)
        .arg(len as u32)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
    }
}

pub(super) fn max_pool2d_backward<S1: DataRef<Elem = f32>, S2: DataMut<Elem = f32>, S3: DataRef<Elem = f32>>(
    input: &TensorBase<S1, Ix4>,
    input_grad: &mut TensorBase<S2, Ix4>,
    args: &Pool2dArgs,
    output_grad: &TensorBase<S3, Ix4>,
) {
    let xpu = input.device.opencl().unwrap();
    
    let x = input.as_ocl_buffer().unwrap();
    let dx = input_grad.as_mut_ocl_buffer().unwrap();
    let dy = output_grad.as_ocl_buffer().unwrap();
    
    let (n, ic, ih, iw) = input.dim();
    let (_n, oc, oh, ow) = output_grad.dim();
    
    let [kh, kw] = args.kernel;
    let [sh, sw] = args.strides;
    let [ph, pw] = args.padding;
    
    let len = n * oc;
    let nthreads = 64;
    let mut nblocks = len / nthreads;
    if len < nthreads || len % nthreads != 0 {
        nblocks += 1;
    }
    
    let kernel = Kernel::builder()
        .name("max_pool2d_backward")
        .program(&xpu.program)
        .queue(xpu.queue.clone())
        .global_work_size(nblocks * nthreads)
        .local_work_size(nthreads)
        .arg(&x)
        .arg(&dx)
        .arg(&dy)
        .arg(ih as u32)
        .arg(iw as u32)
        .arg(oh as u32)
        .arg(ow as u32)
        .arg(kh as u32)
        .arg(kw as u32)
        .arg(sh as u32)
        .arg(sw as u32)
        .arg(ph as u32)
        .arg(pw as u32)
        .arg(len as u32)
        .build()
        .unwrap();
    unsafe {
        kernel.enq().unwrap();
    }
}
