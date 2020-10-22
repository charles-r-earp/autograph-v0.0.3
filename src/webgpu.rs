use super::{DataMut, DataRef, Num, Unsigned, TensorBase, Transpose};
use ndarray::{Dimension, Ix0, Ix1, Ix2};
use futures::executor::LocalPool;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, LockResult};
use std::any::TypeId;
use std::mem::ManuallyDrop;
use wgpu::{
    Adapter, BackendBit, Buffer as WgpuBuffer, BufferSlice as WgpuBufferSlice, BufferUsage, CommandBuffer, Device as WgpuDevice,
    Instance, Queue, ShaderModule, ShaderModuleSource, ComputePipeline, BindGroup, BindGroupLayout, ComputePass, CommandEncoder,
};

mod gemm;
use gemm::{Gemm, Shape2d};

pub trait WebType: bytemuck::Pod + Debug + Display {
    fn shader_type() -> &'static str;
    fn type_name() -> &'static str;
}

impl WebType for u8 {
    fn shader_type() -> &'static str {
        "uint"
    }
    fn type_name() -> &'static str {
        "u8"
    }
}

impl WebType for f32 {
    fn shader_type() -> &'static str {
        "float"
    }
    fn type_name() -> &'static str {
        "f32"
    }
}

struct ComputeDescriptor {
    bind_group_layout: BindGroupLayout,
    compute_pipeline: ComputePipeline,
}

struct ComputeTaskBuilder<'a, F> {
    key: String,
    source_fn: Option<F>,
    slices: &'a [WgpuBufferSlice<'a>],
    push_constants: [u32; 128/4],
    push_constant_size: usize,
    work_groups: [u32; 3],
}   

impl<'a, F: Fn() -> String> ComputeTaskBuilder<'a, F> {
    fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            source_fn: None, 
            slices: &[],
            push_constants: unsafe { std::mem::uninitialized() },
            push_constant_size: 0,
            work_groups: [1, 1, 1],
        }
    }
    fn source(mut self, source: F) -> Self {
        self.source_fn.replace(source);
        self
    }
    fn buffers(mut self, slices: &'a [WgpuBufferSlice<'a>]) -> Self {
        self.slices = slices;
        self
    }
    fn push_constants(mut self, push_constants: &[u32]) -> Self {
        self.push_constants[..push_constants.len()]
            .copy_from_slice(push_constants);
        self.push_constant_size = push_constants.len();
        self
    }
    fn work_groups(mut self, work_groups: [u32; 3]) -> Self {
        self.work_groups = work_groups;
        self
    }
}

struct ComputeTask {
    key: String,
    bind_group: BindGroup,
    push_constants: [u32; 128/4],
    push_constant_size: usize,
    work_groups: [u32; 3],
}

fn shader_to_spirv(
    source: impl AsRef<str>
) -> ShaderModuleSource<'static> {
    use std::io::Read;
    let source = source.as_ref();
    //println!("{}", &source);
    let mut spirv_file = glsl_to_spirv::compile(source, glsl_to_spirv::ShaderType::Compute)
        .unwrap();
    let mut spirv_u8 = Vec::new();
    spirv_file.read_to_end(&mut spirv_u8)
        .unwrap();
    debug_assert_eq!(spirv_u8.len() % 4, 0);
    debug_assert_eq!(spirv_u8.capacity() % 4, 0);
    let spirv_u32 = unsafe {
        Vec::from_raw_parts(
            spirv_u8.as_mut_ptr() as *mut u32,
            spirv_u8.len() / 4,
            spirv_u8.capacity() / 4,
        )
    };
    std::mem::forget(spirv_u8);
    ShaderModuleSource::SpirV(spirv_u32.into())
}

struct BufferCopyBuilder<'a> {
    source: &'a WgpuBuffer,
    source_offset: wgpu::BufferAddress,
    destination: &'a WgpuBuffer,
    destination_offset: wgpu::BufferAddress,
    copy_size: wgpu::BufferAddress
}

impl<'a> BufferCopyBuilder<'a> {
    fn new(source: &'a WgpuBuffer, destination: &'a WgpuBuffer, copy_size: wgpu::BufferAddress) -> Self {
        Self {
            source,
            source_offset: 0,
            destination,
            destination_offset: 0,
            copy_size
        }
    }
}

struct WebGpuBaseGuard<'a> {
    device: &'a WgpuDevice,
    queue: &'a Queue,
    base: MutexGuard<'a, WebGpuBase>
}

impl<'a> WebGpuBaseGuard<'a> {
    fn compute_task(&mut self, task: ComputeTaskBuilder<impl Fn() -> String>) {
        let device = self.device;
        let queue = self.device;
        let base = &mut self.base;
        
        let ComputeTaskBuilder {
            key,
            source_fn,
            slices,
            push_constants,
            push_constant_size,
            work_groups,
        } = task;
        
        let descriptor = base.compute_descriptors.entry(key.clone())
            .or_insert_with(|| {
                let spirv = shader_to_spirv(&source_fn.unwrap()());
                let shader = device.create_shader_module(spirv);
             
                let mut bind_group_layout_entries: [wgpu::BindGroupLayoutEntry; 8] = unsafe { std::mem::uninitialized() };
                (0 .. slices.len())
                    .zip(&mut bind_group_layout_entries)
                    .for_each(|(i, entry)| {
                        *entry = wgpu::BindGroupLayoutEntry {
                            binding: i as _,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                                min_binding_size: wgpu::BufferSize::new(4),
                            },
                            count: None,
                        }
                    });
                
                let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &bind_group_layout_entries[..slices.len()]
                });
                
                let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStage::COMPUTE,
                        range: 0..(push_constant_size * 4) as _,
                    }],
                });

                let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &shader,
                        entry_point: "main",
                    },
                });
             
                ComputeDescriptor {
                    bind_group_layout,
                    compute_pipeline
                }
            });
            
        let mut bind_group_entries: [wgpu::BindGroupEntry; 8] = unsafe { std::mem::uninitialized() };
        slices.iter()
            .zip(bind_group_entries.as_mut())
            .enumerate()
            .for_each(|(i, (slice, bind_group_entry))| {
                *bind_group_entry = wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: wgpu::BindingResource::Buffer(*slice)
                };
            }); 
                
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &descriptor.bind_group_layout,
            entries: &bind_group_entries[..slices.len()],
        });  
        
        base.compute_tasks.push(ComputeTask {
            key,
            bind_group,
            push_constants,
            push_constant_size,
            work_groups,
        });
    }
    fn copy_buffer_to_buffer(&mut self, copy: BufferCopyBuilder) {
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None }
        );
        
        encoder.copy_buffer_to_buffer(
            copy.source,
            copy.source_offset,
            copy.destination,
            copy.destination_offset,
            copy.copy_size
        );
        
        self.synchronize(Some(encoder.finish()));
    }
    fn synchronize(&mut self, copy: Option<CommandBuffer>) {  
        if !self.base.compute_tasks.is_empty() {
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: None }
            );
        
            let tasks = std::mem::replace(&mut self.base.compute_tasks, Vec::new());
            
            {
                let mut cpass = encoder.begin_compute_pass();
                
                tasks.iter()
                    .for_each(|t| {
                        cpass.set_bind_group(0, &t.bind_group, &[]);
                        
                        let descriptor = self.base.compute_descriptors.get(&t.key)
                            .unwrap();
                        cpass.set_pipeline(&descriptor.compute_pipeline); 
                        
                        if t.push_constant_size > 0 {
                            cpass.set_push_constants(0, &t.push_constants[..t.push_constant_size]);
                        }
                        let [wgx, wgy, wgz] = t.work_groups;
                        cpass.dispatch(wgx, wgy, wgz);
                    });
            }
            
            let compute = encoder.finish();
            
            self.queue.submit(
                Some(compute)
                    .into_iter()
                    .chain(copy)
            );
        }
        else {
            if copy.is_some() {
                self.queue.submit(copy);
            }
        };
    
        self.base.buffers.clear();
    } 
}

#[derive(Default)]
struct WebGpuBase {
    compute_descriptors: HashMap<String, ComputeDescriptor>,
    compute_tasks: Vec<ComputeTask>,
    buffers: Vec<WgpuBuffer>,
}

pub struct WebGpu {
    instance: Instance,
    adapter: Adapter,
    index: usize,
    device: WgpuDevice,
    queue: Queue,
    base: Mutex<WebGpuBase>,
}

impl WebGpu {
    pub fn new(index: usize) -> Arc<Self> {
        use futures::task::LocalSpawnExt;
        async fn new_async(index: usize) -> Arc<WebGpu> {
            let backend = BackendBit::all();
            let instance = Instance::new(backend);
            if let Some(adapter) = instance.enumerate_adapters(backend).skip(index).next() {
                let mut limits = wgpu::Limits::default();
                limits.max_push_constant_size = 128;
                let (device, queue) = adapter
                    .request_device(
                        &wgpu::DeviceDescriptor {
                            features: wgpu::Features::PUSH_CONSTANTS
                                /*| wgpu::Features::MAPPABLE_PRIMARY_BUFFERS*/,
                            limits,
                            shader_validation: true,
                        },
                        None,
                    )
                    .await
                    .unwrap();
                let base = Mutex::new(WebGpuBase::default());
                Arc::new(WebGpu {
                    instance,
                    adapter,
                    index,
                    device,
                    queue,
                    base
                })
            } else {
                panic!("No WebGpu Devices!");
            }
        }
        type Output = Option<Arc<WebGpu>>;
        let mut output = None;
        {
            let mut pool = futures::executor::LocalPool::new();
            let spawner = pool.spawner();
            {
                let output: &'static mut () = unsafe { std::mem::transmute(&mut output) };
                spawner
                    .spawn_local(async move {
                        let output: &mut Output = unsafe { std::mem::transmute(output) };
                        output.replace(new_async(index).await);
                    })
                    .unwrap();
            }
            pool.run();
        }
        output.unwrap()
    }
    fn base(&self) -> LockResult<WebGpuBaseGuard> {
        self.base.lock()
            .map(|base| {
                WebGpuBaseGuard {
                    device: &self.device,
                    queue: &self.queue,
                    base
                }
            })
            .map_err(|e| {
                PoisonError::new(WebGpuBaseGuard {
                    device: &self.device,
                    queue: &self.queue,
                    base: e.into_inner()
                })
            })
    }
    pub fn synchronize(&self) {
        self.base()
            .unwrap()
            .synchronize(None);
    }
}

impl Debug for WebGpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "WebGpu({})", self.index)
    }
}

pub struct WebBuffer<T: Num> {
    buffer: Option<WgpuBuffer>,
    len: usize,
    gpu: Arc<WebGpu>,
    _m: PhantomData<T>,
}

impl<T: Num> WebBuffer<T> {
    fn default_usage() -> BufferUsage {
        BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC // | BufferUsage::MAP_READ //| BufferUsage::MAP_WRITE
    }
    pub(super) unsafe fn uninitialized(gpu: &Arc<WebGpu>, len: usize) -> Self {
        let gpu = gpu.clone();
        
        let buffer = if len > 0 {
            let mut size = (len * std::mem::size_of::<T>()) as wgpu::BufferAddress;
            size += wgpu::COPY_BUFFER_ALIGNMENT - (size % wgpu::COPY_BUFFER_ALIGNMENT);
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: Self::default_usage(),
                mapped_at_creation: false,
            });
            Some(buffer)
        }
        else {
            None
        };
        
        Self {
            buffer,
            len,
            gpu,
            _m: <_>::default(),
        }
    }
    pub(super) fn fill(&mut self, elem: T) {
        if let Some(buffer) = &self.buffer {
            let gpu = &self.gpu;
            let device = &gpu.device;
            
            let local_size = 1024;

            let src = include_str!("webgpu/fill/fill.comp");
            let shader_name = format!("fill_{}_{}", T::type_name(), local_size);

            let mut push_constants = [0u32; 1];
            let len = self.len as u32;
            let len = if TypeId::of::<T>() == TypeId::of::<u8>() {
                bytemuck::cast_slice_mut(push_constants.as_mut()).copy_from_slice(&[elem; 4]);
                if len < 4 {
                    1
                }
                else if len % 4 == 0 {
                    len / 4    
                }
                else {
                    len / 4 + 1 
                }
            }
            else {
                bytemuck::cast_slice_mut(push_constants.as_mut())[0] = elem;
                len
            }; 
            
            let work_groups = if len < local_size {
                1
            }
            else if len % local_size == 0 {
                len / local_size
            }
            else {
                len / local_size + 1
            };
            
            let slices = &[buffer.slice(..)];
            
            let task = ComputeTaskBuilder::new(shader_name)
                .source(|| format!(
                    "#version 450\n#define T {}\n#define LOCAL_SIZE {}\n{}", 
                    T::shader_type(),
                    local_size,
                    src
                ))
                .buffers(slices)
                .push_constants(&push_constants)
                .work_groups([work_groups, 1, 1]);
                
            
            gpu.base()
                .unwrap()
                .compute_task(task);
        }
    }
    pub(super) fn zeros(gpu: &Arc<WebGpu>, len: usize) -> Self {
        Self::from_elem(gpu, T::zero(), len)
    }
    pub(super) fn ones(gpu: &Arc<WebGpu>, len: usize) -> Self {
        Self::from_elem(gpu, T::one(), len)
    }
    pub(super) fn from_elem(gpu: &Arc<WebGpu>, elem: T, len: usize) -> Self {
        let mut buffer = unsafe { Self::uninitialized(gpu, len) };

        buffer.fill(elem);

        buffer
    }
    pub(super) fn from_slice(gpu: &Arc<WebGpu>, slice: &[T]) -> Self {
        use wgpu::util::DeviceExt;
        let gpu = gpu.clone();
        
        let len = slice.len();
        
        let buffer = if len > 0 {
            let buffer = gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(slice),
                    usage: Self::default_usage(),
                });
            Some(buffer)
        }
        else {
            None
        };

        Self {
            buffer,
            len,
            gpu,
            _m: <_>::default(),
        }
    }
    pub(super) fn copy_from_slice(&mut self, slice: &[T]) {
        unimplemented!();
        
        /*if let Some(buffer) = &self.buffer {
            let gpu = &self.gpu;
            gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(slice));
        }
        else {
            debug_assert_eq!(slice.len(), 0);
        }*/
        /*
        use futures::task::LocalSpawnExt;
                
        let gpu = &self.gpu;
        
        gpu.queue.submit(None);
        
        let buffer_slice = self.buffer.slice(..);
        
        let map_future = buffer_slice.map_async(wgpu::MapMode::Write);
        
        gpu.device.poll(wgpu::Maintain::Wait);
        
        let mut pool = LocalPool::new();
        let spawner = pool.spawner();
        spawner
            .spawn_local(async {
                map_future.await.unwrap();
            })
            .unwrap();
        pool.run();

        {
            let mut data = buffer_slice.get_mapped_range_mut();
            bytemuck::cast_slice_mut(&mut *data)[..slice.len()].copy_from_slice(slice);
        };
        
        self.buffer.unmap();
        */
    }
    pub(super) fn to_vec(&self) -> Vec<T> {
        if let Some(buffer) = &self.buffer {
            
            let gpu = &self.gpu;
            
            let size = (self.len * std::mem::size_of::<T>()) as wgpu::BufferAddress;
            let size = size + wgpu::COPY_BUFFER_ALIGNMENT - (size % wgpu::COPY_BUFFER_ALIGNMENT);
            let tmp_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                mapped_at_creation: false,
            });
            
            gpu.base()
                .unwrap()
                .copy_buffer_to_buffer(
                    BufferCopyBuilder::new(
                        buffer, &tmp_buffer, size
                    ) 
                );
            
            let slice = tmp_buffer.slice(..);
            let slice_future = slice.map_async(wgpu::MapMode::Read);
            
            gpu.device.poll(wgpu::Maintain::Wait);
            
            let vec = bytemuck::cast_slice(&slice.get_mapped_range())[..self.len]
                    .to_vec();
            
            tmp_buffer.unmap();
            
            vec
        }
        else {
            Vec::new()
        }
    }
    fn slice(&self, bounds: impl std::ops::RangeBounds<wgpu::BufferAddress>) -> WgpuBufferSlice {
        self.buffer.as_ref()
            .unwrap()
            .slice(..)
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
}

impl<T: Num> Clone for WebBuffer<T> {
    fn clone(&self) -> Self {
        unimplemented!()
    }
}

/*
impl<T: Num> Drop for WebBuffer<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // move buffer into gpu_base
        }
    }
}*/


fn cast<T1: Num, S1: DataRef<Elem=T1>, T2: Num, S2: DataMut<Elem=T2>, D: Dimension>(input: &TensorBase<S1, D>, output: &mut TensorBase<S2, D>) {
    use num_traits::Bounded;
    
    let gpu = input.device().web().unwrap();
    let device = &gpu.device;
    
    let len = input.len() as u32;
    let nclasses = 1;
    
    let local_size = 1024;
    
    let work_groups = if len < local_size {
        1
    } else if len % local_size == 0 {
        len / local_size
    } else {
        len / local_size + 1
    };
    
    let x = input.as_web_buffer().unwrap();
    let mut y = output.as_mut_web_buffer().unwrap();
    
    let push_constants = [len, nclasses];
    let push_constants = push_constants.as_ref();
    
    let src = include_str!("webgpu/cast/cast.comp");
    let shader_name = format!("cast_{}_{}_{}", T1::type_name(), T2::type_name(), local_size);
    
    let slices = &[x.slice(..), y.slice(..)];
    
    let task = ComputeTaskBuilder::new(shader_name)
        .source(|| {
            let t1_byte = if TypeId::of::<T1>() == TypeId::of::<u8>() {
                "#define T1_BYTE\n"
            }
            else {
                ""
            };
            format!(
                "#version 450\n#define T1 {}\n#define T2 {}\n#define SCALE 1/T2({})\n#define LOCAL_SIZE {}\n{}{}",
                T1::shader_type(),
                T2::shader_type(),
                T1::max_value(),
                local_size,
                t1_byte,
                src
            )
        })
        .buffers(slices)
        .push_constants(push_constants)
        .work_groups([work_groups, 1, 1]);
    
    gpu.base()
        .unwrap()
        .compute_task(task);
}

fn cast_one_hot<T1: Unsigned, S1: DataRef<Elem=T1>, T2: Num, S2: DataMut<Elem=T2>>(input: &TensorBase<S1, Ix1>, output: &mut TensorBase<S2, Ix2>) {
    use num_traits::Bounded;
    
    let gpu = input.device().web().unwrap();
    let device = &gpu.device;
    
    let len = input.len() as u32;
    let nclasses = output.raw_dim()[1] as u32;
    
    let local_size = 1024;
    
    let work_groups = if len < local_size {
        1
    } else if len % local_size == 0 {
        len / local_size
    } else {
        len / local_size + 1
    };
    
    let x = input.as_web_buffer().unwrap();
    let mut y = output.as_mut_web_buffer().unwrap();
    
    let push_constants = [len, nclasses];
    let push_constants = push_constants.as_ref();
    
    let src = include_str!("webgpu/cast/cast.comp");
    let shader_name = format!("cast_one_hot_{}_{}_{}", T1::type_name(), T2::type_name(), local_size);
    
    let slices = &[x.slice(..), y.slice(..)];
    
    let task = ComputeTaskBuilder::new(shader_name)
        .source(|| {
            let t1_byte = if TypeId::of::<T1>() == TypeId::of::<u8>() {
                "#define T1_BYTE\n"
            }
            else {
                ""
            };
            format!(
                "#version 450\n#define T1 {}\n#define T2 {}\n#define ONE_HOT\n#define SCALE 1/T2({})\n#define LOCAL_SIZE {}\n{}{}",
                T1::shader_type(),
                T2::shader_type(),
                T1::max_value(),
                local_size,
                t1_byte,
                src
            )
        })
        .buffers(slices)
        .push_constants(push_constants)
        .work_groups([work_groups, 1, 1]);
    
    gpu.base()
        .unwrap()
        .compute_task(task);
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
    cast(input, output);
}

pub(super) fn unsigned_to_one_hot_f32<
    T: Unsigned,
    S1: DataRef<Elem = T>,
    S2: DataMut<Elem = f32>,
>(
    input: &TensorBase<S1, Ix1>,
    output: &mut TensorBase<S2, Ix2>,
) {
    cast_one_hot(input, output);
}

#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct AxpyPushConsts<T: Num> {
    n: u32,
    alpha: T,
    offx: u32,
    incx: i32,
    offy: u32,
    incy: i32,
}

unsafe impl<T: Num> bytemuck::Zeroable for AxpyPushConsts<T> {}

unsafe impl<T: Num> bytemuck::Pod for AxpyPushConsts<T> {}

fn axpy<T: Num>(
    n: u32,
    alpha: T,
    x: &WebBuffer<T>,
    offx: u32,
    incx: i32,
    y: &mut WebBuffer<f32>,
    offy: u32,
    incy: i32,
) {
    let gpu = &x.gpu;

    let local_size = 1024;

    let work_groups = if n < local_size {
        1
    } else if n % local_size == 0 {
        n / local_size
    } else {
        n / local_size + 1
    };

    let src = include_str!("webgpu/axpy/axpy.comp");
    let shader_name = format!("axpy_{}_{}_{}", T::type_name(), n, local_size);

    let push_constants = &[AxpyPushConsts {
        n,
        alpha,
        offx,
        incx,
        offy,
        incy,
    }];
    let push_constants = bytemuck::cast_slice(push_constants);
    
    let slices = &[x.slice(..), y.slice(..)];
            
    let task = ComputeTaskBuilder::new(shader_name)
        .source(|| format!(
            "#version 450\n#define T {}\n#define N {}\n#define LOCAL_SIZE {}\n{}", 
            T::shader_type(),
            n,
            local_size,
            src
        ))
        .buffers(slices)
        .push_constants(&push_constants)
        .work_groups([work_groups, 1, 1]);
        
    
    gpu.base()
        .unwrap()
        .compute_task(task);
}

pub(super) fn scaled_add<S1: DataMut<Elem = f32>, S2: DataRef<Elem = f32>, D: Dimension>(
    lhs: &mut TensorBase<S1, D>,
    alpha: f32,
    rhs: &TensorBase<S2, D>,
) {
    let n = lhs.len() as u32;
    let x = rhs.as_web_buffer().unwrap();
    let y = lhs.as_mut_web_buffer().unwrap();
    axpy(n, alpha, x, 0, 1, y, 0, 1);
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
    let a_shape = {
        let (d1, d2) = a.dim();
        let a_shape = Shape2d::new([d1 as u32, d2 as u32]);
        match trans_a {
            Transpose::No => a_shape,
            Transpose::Yes => a_shape.t(),
        }
    };
    let b_shape = {
        let (d1, d2) = b.dim();
        let b_shape = Shape2d::new([d1 as u32, d2 as u32]);
        match trans_b {
            Transpose::No => b_shape,
            Transpose::Yes => b_shape.t(),
        }
    };
    let c_shape = {
        let (d1, d2) = c.dim();
        Shape2d::new([d1 as u32, d2 as u32])
    };

    Gemm::new(
        a.as_web_buffer().unwrap(),
        b.as_web_buffer().unwrap(),
        c.as_mut_web_buffer().unwrap(),
    )
        .alpha(alpha)
        .a_shape(a_shape)
        .b_shape(b_shape)
        .beta(beta)
        .c_shape(c_shape)
        .exec_v1();
}

pub(super) fn broadcast<T: Num, S1: DataRef<Elem = T>, S2: DataMut<Elem = T>, D: Dimension>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, D::Larger>,
) { 
    let gpu = input.device().web().unwrap();
    let device = &gpu.device;
    
    let nclasses = input.len() as u32;
    let len = output.len() as u32; 
    
    let local_size = 1024;
    
    let work_groups = if len < local_size {
        1
    } else if len % local_size == 0 {
        len / local_size
    } else {
        len / local_size + 1
    };
    
    let x = input.as_web_buffer().unwrap();
    let mut y = output.as_mut_web_buffer().unwrap();
    
    let push_constants = [len, nclasses];
    let push_constants = push_constants.as_ref();
    
    let src = include_str!("webgpu/broadcast/broadcast.comp");
    let shader_name = format!("broadcast_{}_{}", T::type_name(), local_size);
    
    let slices = &[x.slice(..), y.slice(..)];
    
    let task = ComputeTaskBuilder::new(&shader_name)
        .source(|| format!(
            "#version 450\n#define T {}\n#define LOCAL_SIZE {}\n{}",
            T::shader_type(),
            local_size,
            src
        ))
        .buffers(slices)
        .push_constants(push_constants)
        .work_groups([work_groups, 1, 1]);
        
    gpu.base()
        .unwrap()
        .compute_task(task);
}

pub(super) fn broadcast_backward<T: Num, S1: DataMut<Elem = T>, S2: DataRef<Elem = T>, D: Dimension>(
    input_grad: &mut TensorBase<S1, D>,
    output_grad: &TensorBase<S2, D::Larger>,
) {
    let gpu = output_grad.device().web().unwrap();
    let device = &gpu.device;
    
    let len = input_grad.len() as u32;
    let nclasses = output_grad.raw_dim().slice()[1..].iter().product::<usize>() as u32;
    
    let local_size = 1024;
    
    let work_groups = if len < local_size {
        1
    } else if len % local_size == 0 {
        len / local_size
    } else {
        len / local_size + 1
    };
    
    let mut dx = input_grad.as_mut_web_buffer().unwrap();
    let dy = output_grad.as_web_buffer().unwrap();
    
    let push_constants = [len];
    let push_constants = push_constants.as_ref();
    
    let src = include_str!("webgpu/broadcast/broadcast_backward.comp");
    let shader_name = format!("broadcast_backward_{}_{}_{}", T::type_name(), nclasses, local_size);
    
    
    let slices = &[dx.slice(..), dy.slice(..)];
    
    let task = ComputeTaskBuilder::new(&shader_name)
        .source(|| format!(
            "#version 450\n#define T {}\n#define C {}\n#define LOCAL_SIZE {}\n{}",
            T::shader_type(),
            nclasses,
            local_size,
            src
        ))
        .buffers(slices)
        .push_constants(push_constants)
        .work_groups([work_groups, 1, 1]);
        
    gpu.base()
        .unwrap()
        .compute_task(task);
}

pub(super) fn reduce_sum<T: Num, S1: DataRef<Elem = T>, S2: DataMut<Elem = T>, D: Dimension>(
    input: &TensorBase<S1, D>,
    output: &mut TensorBase<S2, Ix0>,
) {
    let gpu = input.device.web().unwrap();
    let device = &gpu.device;
    
    let len = input.len() as u32;
    let local_size = 256;
    
    let x = input.as_web_buffer().unwrap();
    let mut tmp: WebBuffer<T>;
    let mut y = output.as_mut_web_buffer().unwrap();
    
    let mut base = gpu.base()
        .unwrap();
    
    let x = if len > (2 * local_size) {
        let work_groups = if len % (2 * local_size) == 0 {
            len / (2 * local_size)
        }
        else {
            len / (2 * local_size) + 1
        };
        
        tmp = unsafe { 
            WebBuffer::uninitialized(&gpu, work_groups as usize)
        };
        
        let push_constants = [len];
        let push_constants = push_constants.as_ref();
        
        let src = include_str!("webgpu/reduce/reduce_partial.comp");
        let shader_name = format!("reduce_sum_partial_{}_{}", T::type_name(), local_size);
        
        let slices = &[x.slice(..), tmp.slice(..)];
    
        let partial_sum = ComputeTaskBuilder::new(&shader_name)
            .source(|| format!(
                "#version 450\n#define T {}\n#define LOCAL_SIZE {}\n{}",
                T::shader_type(),
                local_size,
                src
            ))
            .buffers(slices)
            .push_constants(push_constants)
            .work_groups([work_groups, 1, 1]);
            
        base.compute_task(partial_sum);
        
        &tmp
    } else {
        x
    };
    
    {
        let n = x.len() as u32;
        
        let src = include_str!("webgpu/reduce/reduce_final.comp");
        let shader_name = format!("reduce_sum_final_{}_{}", T::type_name(), n);
        
        let slices = &[x.slice(..), y.slice(..)];
    
        let final_sum = ComputeTaskBuilder::new(&shader_name)
            .source(|| format!(
                "#version 450\n#define T {}\n#define N {}\n{}",
                T::shader_type(),
                n,
                src
            ))
            .buffers(slices);
        
        base.compute_task(final_sum);
    }
}

pub(super) fn cross_entropy<
    T: Num,
    S1: DataRef<Elem = T>,
    S2: DataRef<Elem = T>,
    S3: DataMut<Elem = T>,
>(
    input: &TensorBase<S1, Ix2>,
    target: &TensorBase<S2, Ix2>,
    output: &mut TensorBase<S3, Ix2>,
) {
    let gpu = input.device().web().unwrap();
    let device = &gpu.device;
    
    let (batch_size, nclasses) = input.dim();
    
    let local_size = 1024;
    
    let m = batch_size as u32;
    let n = nclasses as u32;
    
    let work_groups = if m < local_size {
        1
    } else if m % local_size == 0 {
        m / local_size
    } else {
        m / local_size + 1
    };
    
    let x = input.as_web_buffer().unwrap();
    let t = target.as_web_buffer().unwrap();
    let mut y = output.as_mut_web_buffer().unwrap();
    
    let push_constants = [m];
    let push_constants = push_constants.as_ref();
    
    let src = include_str!("webgpu/cross_entropy/cross_entropy.comp");
    let shader_name = format!("cross_entropy_{}_{}_{}", T::type_name(), n, local_size);
    
    let slices = &[x.slice(..), t.slice(..), y.slice(..)];
    
    let task = ComputeTaskBuilder::new(&shader_name)
        .source(|| format!(
            "#version 450\n#define T {}\n#define C {}\n#define LOCAL_SIZE {}\n{}",
            T::shader_type(),
            n,
            local_size,
            src
        ))
        .buffers(slices)
        .push_constants(push_constants)
        .work_groups([work_groups, 1, 1]);
        
    gpu.base()
        .unwrap()
        .compute_task(task);
}

pub(super) fn cross_entropy_backward<
    T: Num,
    S1: DataRef<Elem = T>,
    S2: DataMut<Elem = T>,
    S3: DataRef<Elem = T>,
    S4: DataRef<Elem = T>,
>(
    input: &TensorBase<S1, Ix2>,
    input_grad: &mut TensorBase<S2, Ix2>,
    target: &TensorBase<S3, Ix2>,
    output_grad: &TensorBase<S4, Ix0>,
) {
    let gpu = input.device().web().unwrap();
    let device = &gpu.device;
    
    let len = input.len() as u32;
    
    let local_size = 1024;
    
    let work_groups = if len < local_size {
        1
    } else if len % local_size == 0 {
        len / local_size
    } else {
        len / local_size + 1
    };
    
    let x = input.as_web_buffer().unwrap();
    let dx = input_grad.as_mut_web_buffer().unwrap();
    let t = target.as_web_buffer().unwrap();
    let dy = output_grad.as_web_buffer().unwrap();
    
    let push_constants = [len];
    let push_constants = push_constants.as_ref();
    
    let src = include_str!("webgpu/cross_entropy/cross_entropy_backward.comp");
    let shader_name = format!("cross_entropy_backward_{}_{}", T::type_name(), local_size);
    
    let slices = &[x.slice(..), dx.slice(..), t.slice(..), dy.slice(..)];
    
    let task = ComputeTaskBuilder::new(&shader_name)
        .source(|| format!(
            "#version 450\n#define T {}\n#define LOCAL_SIZE {}\n{}",
            T::shader_type(),
            local_size,
            src
        ))
        .buffers(slices)
        .push_constants(push_constants)
        .work_groups([work_groups, 1, 1]);
        
    gpu.base()
        .unwrap()
        .compute_task(task);
}
