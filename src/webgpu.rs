use super::{DataMut, DataRef, Ix2, Num, TensorBase, Transpose};
use futures::executor::LocalPool;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};
use wgpu::{
    Adapter, BackendBit, Buffer as WgpuBuffer, BufferUsage, CommandBuffer, Device as WgpuDevice,
    Instance, Queue, ShaderModule, ShaderModuleSource,
};
use wgpu::{BindGroup, ComputePipeline};

mod gemm;
use gemm::{Gemm, Shape2d};

pub trait WebType: bytemuck::Pod {
    fn shader_type() -> &'static str;
    fn type_name() -> &'static str;
}

impl WebType for u8 {
    fn shader_type() -> &'static str {
        "unsigned char"
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
                                | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
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
    pub fn synchronize(&self) {}
}

impl Debug for WebGpu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "WebGpu({})", self.index)
    }
}

pub struct WebBuffer<T: Num> {
    buffer: WgpuBuffer,
    len: usize,
    gpu: Arc<WebGpu>,
    _m: PhantomData<T>,
}

impl<T: Num> WebBuffer<T> {
    fn default_usage() -> BufferUsage {
        BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC | BufferUsage::MAP_READ
    }
    pub(super) unsafe fn uninitialized(gpu: &Arc<WebGpu>, len: usize) -> Self {
        let gpu = gpu.clone();
        let size = (len * std::mem::size_of::<T>()) as wgpu::BufferAddress;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: Self::default_usage(),
            mapped_at_creation: false,
        });
        Self {
            buffer,
            len,
            gpu,
            _m: <_>::default(),
        }
    }
    pub(super) fn fill(&mut self, elem: T) {
        let gpu = &self.gpu;
        let device = &gpu.device;

        let src = include_str!("webgpu/fill/fill.comp");
        let shader_name = format!("fill_{}", T::type_name());

        let mut push_constants = [0u32; 1];
        bytemuck::cast_slice_mut(push_constants.as_mut())[0] = elem;

        let shader = gpu
            .shader(&shader_name, || {
                let mut source = String::from("#version 450\n");
                source.push_str(&format!("#define T {}\n", T::shader_type()));
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
            cpass.dispatch(self.len as _, 1, 1);
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
        let gpu = gpu.clone();
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

        gpu.device.poll(wgpu::Maintain::Wait);

        Self {
            buffer,
            len,
            gpu,
            _m: <_>::default(),
        }
    }
    pub(super) fn copy_from_slice(&mut self, slice: &[T]) {
        unimplemented!();
    }
    pub(super) fn to_vec(&self) -> Vec<T> {
        use futures::task::LocalSpawnExt;
        let gpu = &self.gpu;

        gpu.queue.submit(None);

        let slice = self.buffer.slice(..);
        let slice_future = slice.map_async(wgpu::MapMode::Read);

        gpu.device.poll(wgpu::Maintain::Wait);

        let mut pool = LocalPool::new();
        let spawner = pool.spawner();
        spawner
            .spawn_local(async {
                slice_future.await.unwrap();
            })
            .unwrap();
        pool.run();

        let vec = {
            let data = slice.get_mapped_range();
            let data_slice =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, self.len) };
            data_slice.to_vec()
        };
        self.buffer.unmap();
        vec
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
pub fn matmul(m: usize, k: usize, n: usize, a: &WebBuffer<f32>, b: &WebBuffer<f32>, c: &mut WebBuffer<f32>) {
    let [m, k, n] = [m as _, k as _, n as _];
    let gpu = &a.gpu;
    Gemm::new(a, b, c)
        .a_shape(Shape2d::new([m, k]))
        .b_shape(Shape2d::new([k, n]))
        .c_shape(Shape2d::new([m, n]))
        .exec_v1()
        .unwrap();
}*/

#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct AxpyPushConsts<T: Num> {
    alpha: T,
    offx: u32,
    incx: i32,
    offy: u32,
    incy: i32,
}

unsafe impl<T: Num> bytemuck::Zeroable for AxpyPushConsts<T> {}

unsafe impl<T: Num> bytemuck::Pod for AxpyPushConsts<T> {}

pub fn axpy<T: Num>(
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
            println!("{}", &source);
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
