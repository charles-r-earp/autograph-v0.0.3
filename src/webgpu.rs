use super::{DataMut, DataRef, Num, Unsigned, TensorBase, Transpose};
use ndarray::{Dimension, Ix0, Ix1, Ix2};
use futures::executor::LocalPool;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};
use std::any::TypeId;
use std::mem::ManuallyDrop;
use wgpu::{
    Adapter, BackendBit, Buffer as WgpuBuffer, BufferUsage, CommandBuffer, Device as WgpuDevice,
    Instance, Queue, ShaderModule, ShaderModuleSource,
};
use wgpu::{BindGroup, ComputePipeline};

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

struct ShaderReadGuard<'a> {
    shaders: RwLockReadGuard<'a, HashMap<String, ShaderModule>>,
    shader: &'a ShaderModule,
}

impl<'a> Deref for ShaderReadGuard<'a> {
    type Target = ShaderModule;
    fn deref(&self) -> &ShaderModule {
        &*self.shader
    }
}

fn shader_to_spirv(
    source: impl AsRef<str>,
    shader_type: glsl_to_spirv::ShaderType,
) -> Result<ShaderModuleSource<'static>, Box<dyn Error>> {
    use std::io::Read;
    let mut spirv_file = glsl_to_spirv::compile(source.as_ref(), shader_type)?;
    let mut spirv_u8 = Vec::new();
    spirv_file.read_to_end(&mut spirv_u8)?;
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
    Ok(ShaderModuleSource::SpirV(spirv_u32.into()))
}

pub struct WebGpu {
    instance: Instance,
    adapter: Adapter,
    device: WgpuDevice,
    index: usize,
    queue: Queue,
    shaders: RwLock<HashMap<String, ShaderModule>>,
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
                let shaders = RwLock::new(HashMap::new());
                Arc::new(WebGpu {
                    instance,
                    adapter,
                    device,
                    index,
                    queue,
                    shaders,
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
    fn shader(
        &self,
        key: impl AsRef<str>,
        mut f: impl FnMut() -> Result<ShaderModule, Box<dyn Error>>,
    ) -> Result<impl Deref<Target = ShaderModule> + '_, Box<dyn Error>> {
        let key: String = key.as_ref().into();
        let shaders = self.shaders.read().unwrap();
        if let Some(shader) = shaders.get(&key) {
            let shader = unsafe { std::mem::transmute(shader) };
            Ok(ShaderReadGuard { shaders, shader })
        } else {
            std::mem::drop(shaders);
            {
                let mut shaders = self.shaders.write().unwrap();
                shaders.insert(key.clone(), f()?);
            }
            let shaders = self.shaders.read().unwrap();
            let shader = shaders.get(&key).unwrap();
            let shader = unsafe { std::mem::transmute(shader) };
            Ok(ShaderReadGuard { shaders, shader })
        }
    }
    pub fn synchronize(&self) {
        self.queue.submit(None);
    }
}

impl Debug for WebGpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "WebGpu({})", self.index)
    }
}

pub struct WebBuffer<T: Num> {
    buffer: ManuallyDrop<WgpuBuffer>,
    len: usize,
    gpu: Arc<WebGpu>,
    _m: PhantomData<T>,
}

impl<T: Num> WebBuffer<T> {
    fn default_usage() -> BufferUsage {
        BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC // | BufferUsage::MAP_READ //| BufferUsage::MAP_WRITE
    }
    pub /*(super)*/ unsafe fn uninitialized(gpu: &Arc<WebGpu>, len: usize) -> Self {
        let gpu = gpu.clone();
        
        let mut size = (len * std::mem::size_of::<T>()) as wgpu::BufferAddress;
        size += wgpu::COPY_BUFFER_ALIGNMENT - (size % wgpu::COPY_BUFFER_ALIGNMENT);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: Self::default_usage(),
            mapped_at_creation: false,
        });
        
        gpu.queue.submit(None);
        
        Self {
            buffer: ManuallyDrop::new(buffer),
            len,
            gpu,
            _m: <_>::default(),
        }
    }
    pub(super) fn fill(&mut self, elem: T) {
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
        
        let shader = gpu
            .shader(&shader_name, || {
                let mut source = String::from("#version 450\n");
                source.push_str(&format!("#define T {}\n", T::shader_type()));
                source.push_str(&format!("#define LOCAL_SIZE {}\n", local_size));
                source.push_str(src);
                let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
                Ok(device.create_shader_module(spirv))
            })
            .unwrap();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(self.buffer.slice(..)),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::COMPUTE,
                range: 0..32,
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

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_push_constants(0, &push_constants); 
            cpass.dispatch(work_groups, 1, 1);
        }

        gpu.queue.submit(Some(encoder.finish()));
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

        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(slice),
                usage: Self::default_usage(),
            });
        
        gpu.queue.submit(None);

        Self {
            buffer: ManuallyDrop::new(buffer),
            len,
            gpu,
            _m: <_>::default(),
        }
    }
    pub(super) fn copy_from_slice(&mut self, slice: &[T]) {
        let gpu = &self.gpu;
        
        gpu.queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(slice));
        
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
        use futures::task::LocalSpawnExt;
        
        let gpu = &self.gpu;
        
        let mut size = (self.len * std::mem::size_of::<T>()) as wgpu::BufferAddress;
        size += wgpu::COPY_BUFFER_ALIGNMENT - (size % wgpu::COPY_BUFFER_ALIGNMENT);
        let tmp_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            mapped_at_creation: false,
        });
        
        //gpu.queue.submit(None);
        
        let mut encoder = gpu.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &tmp_buffer,
            0,
            size
        );
        
        gpu.queue.submit(Some(encoder.finish()));
        
        let slice = tmp_buffer.slice(..);
        let slice_future = slice.map_async(wgpu::MapMode::Read);
        
        gpu.device.poll(wgpu::Maintain::Wait);

        /*{
            let mut pool = LocalPool::new();
            let spawner = pool.spawner();
            spawner
                .spawn_local(async {
                    slice_future.await.unwrap();
                })
                .unwrap();
            pool.run();
        }*/
        
        let vec = bytemuck::cast_slice(&slice.get_mapped_range())[..self.len]
                .to_vec();
        
        tmp_buffer.unmap();
        
        vec
        /*
        use futures::task::LocalSpawnExt;
        
        let gpu = &self.gpu;
        
        let slice = self.buffer.slice(..);
        let slice_future = slice.map_async(wgpu::MapMode::Read);

        let mut pool = LocalPool::new();
        let spawner = pool.spawner();
        spawner
            .spawn_local(async {
                slice_future.await.unwrap();
            })
            .unwrap();
        pool.run();
        
        let vec = bytemuck::cast_slice(&slice.get_mapped_range())[..self.len]
                .to_vec();
        
        self.buffer.unmap();
        vec*/
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

impl<T: Num> Drop for WebBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.buffer);
        }
        self.gpu.queue.submit(None);
    }
}

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
    
    let shader = gpu
        .shader(&shader_name, || {
            let mut source = String::from("#version 450\n");
            source.push_str(&format!("#define T1 {}\n", T1::shader_type()));
            source.push_str(&format!("#define T2 {}\n", T2::shader_type()));
            source.push_str(&format!("#define SCALE 1/T2({})\n", T1::max_value()));
            source.push_str(&format!("#define LOCAL_SIZE {}\n", local_size));
            if (TypeId::of::<T1>() == TypeId::of::<u8>()) {
                source.push_str("#define T1_BYTE");
            }
            source.push_str(src);
            let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
            Ok(device.create_shader_module(spirv))
        })
        .unwrap();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(x.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(y.buffer.slice(..)),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..4*2,
        }],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &shader,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_push_constants(0, &push_constants);
        cpass.dispatch(work_groups, 1, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
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
    
    let shader = gpu
        .shader(&shader_name, || {
            let mut source = String::from("#version 450\n");
            source.push_str(&format!("#define T1 {}\n", T1::shader_type()));
            source.push_str(&format!("#define T2 {}\n", T2::shader_type()));
            source.push_str("#define ONE_HOT\n");
            source.push_str(&format!("#define LOCAL_SIZE {}\n", local_size));
            if (TypeId::of::<T1>() == TypeId::of::<u8>()) {
                source.push_str("#define T1_BYTE");
            }
            source.push_str(src);
            let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
            Ok(device.create_shader_module(spirv))
        })
        .unwrap();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(x.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(y.buffer.slice(..)),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..4*2,
        }],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &shader,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_push_constants(0, &push_constants);
        cpass.dispatch(work_groups, 1, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
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
    let device = &gpu.device;

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

    let shader = gpu
        .shader(&shader_name, || {
            let mut source = String::from("#version 450\n");
            source.push_str(&format!("#define T {}\n", T::shader_type()));
            source.push_str(&format!("#define N {}\n", n));
            source.push_str(&format!("#define LOCAL_SIZE {}\n", local_size));
            source.push_str(src);
            let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
            Ok(device.create_shader_module(spirv))
        })
        .unwrap();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(x.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(y.buffer.slice(..)),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..(4 * push_constants.len()) as u32,
        }],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &shader,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_push_constants(0, &push_constants);
        cpass.dispatch(work_groups, 1, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
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
    
    let shader = gpu
        .shader(&shader_name, || {
            let mut source = String::from("#version 450\n");
            source.push_str(&format!("#define T {}\n", T::shader_type()));
            source.push_str(&format!("#define LOCAL_SIZE {}\n", local_size));
            source.push_str(src);
            let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
            Ok(device.create_shader_module(spirv))
        })
        .unwrap();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            }
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(x.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(y.buffer.slice(..)),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..4*2,
        }],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &shader,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_push_constants(0, &push_constants);
        cpass.dispatch(work_groups, 1, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
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
    
    let shader = gpu
        .shader(&shader_name, || {
            let mut source = String::from("#version 450\n");
            source.push_str(&format!("#define T {}\n", T::shader_type()));
            source.push_str(&format!("#define C {}\n", nclasses));
            source.push_str(&format!("#define LOCAL_SIZE {}\n", local_size));
            source.push_str(src);
            let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
            Ok(device.create_shader_module(spirv))
        })
        .unwrap();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            }
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(dx.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(dy.buffer.slice(..)),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..4,
        }],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &shader,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_push_constants(0, &push_constants);
        cpass.dispatch(work_groups, 1, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
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
    
    let (x, command_partial) = if len > (2 * local_size) {
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
        
        let shader = gpu
            .shader(&shader_name, || {
                let mut source = String::from("#version 450\n");
                source.push_str(&format!("#define T {}\n", T::shader_type()));
                source.push_str(&format!("#define LOCAL_SIZE {}\n", local_size));
                source.push_str(src);
                let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
                Ok(device.create_shader_module(spirv))
            })
            .unwrap();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(x.buffer.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(tmp.buffer.slice(..)),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::COMPUTE,
                range: 0..4,
            }],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &shader,
                entry_point: "main",
            },
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_push_constants(0, &push_constants);
            cpass.dispatch(work_groups, 1, 1);
        }
        (&tmp, Some(encoder.finish()))
        
    } else {
        (x, None)
    };
    
    let command_final = {
        let n = x.len() as u32;
        
        let src = include_str!("webgpu/reduce/reduce_final.comp");
        let shader_name = format!("reduce_sum_final_{}_{}", T::type_name(), n);
        
        let shader = gpu
            .shader(&shader_name, || {
                let mut source = String::from("#version 450\n");
                source.push_str(&format!("#define T {}\n", T::shader_type()));
                source.push_str(&format!("#define N {}\n", n));
                source.push_str(src);
                let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
                Ok(device.create_shader_module(spirv))
            })
            .unwrap();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(x.buffer.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(y.buffer.slice(..)),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &shader,
                entry_point: "main",
            },
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(1, 1, 1);
        }
        encoder.finish()
    };
    
    if let Some(command_partial) = command_partial {
        gpu.queue.submit(Some(command_partial).into_iter().chain(Some(command_final)));
    }
    else {
        gpu.queue.submit(Some(command_final));
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
    
    let shader = gpu
        .shader(&shader_name, || {
            let mut source = String::from("#version 450\n");
            source.push_str(&format!("#define T {}\n", T::shader_type()));
            source.push_str(&format!("#define C {}\n", n));
            source.push_str(&format!("#define LOCAL_SIZE {}\n", local_size));
            source.push_str(src);
            let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
            Ok(device.create_shader_module(spirv))
        })
        .unwrap();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(x.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(t.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(y.buffer.slice(..)),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..4,
        }],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &shader,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_push_constants(0, &push_constants);
        cpass.dispatch(work_groups, 1, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
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
    
    let shader = gpu
        .shader(&shader_name, || {
            let mut source = String::from("#version 450\n");
            source.push_str(&format!("#define T {}\n", T::shader_type()));
            source.push_str(&format!("#define LOCAL_SIZE {}\n", local_size));
            source.push_str(src);
            let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
            Ok(device.create_shader_module(spirv))
        })
        .unwrap();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(x.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(dx.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(t.buffer.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(dy.buffer.slice(..)),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..4,
        }],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &shader,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_push_constants(0, &push_constants);
        cpass.dispatch(work_groups, 1, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));

}
