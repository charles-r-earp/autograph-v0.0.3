use crate::Num;
use super::{WebBuffer, shader_to_spirv};
use bytemuck::{Pod, Zeroable};

use std::error::Error;

#[derive(Default)]
pub(super) struct Shape2d {
    dim: [u32; 2],
    offset: u32,
    strides: [i32; 2],   
}

impl Shape2d {
    pub(super) fn new(dim: [u32; 2]) -> Self {
        let offset = 0;
        let strides = [dim[1] as i32, 1];
        Self {
            dim,
            offset,
            strides,
        }
    } 
    pub(super) fn t(mut self) -> Self {
        let dim = [self.dim[1], self.dim[0]];
        self.dim = dim;
        self.strides = [1, dim[0] as i32];
        self
    }
    pub(super) fn strides(mut self, strides: [i32; 2]) -> Self {
        self.strides = strides;
        self
    }
    pub(super) fn offset(mut self, offset: u32) -> Self {
        self.offset = offset;
        self
    }
}

pub(super) struct Gemm<'a, 'b, 'c, T: Num> {
    alpha: T,
    a: &'a WebBuffer<T>,
    a_shape: Shape2d,
    b: &'b WebBuffer<T>,
    b_shape: Shape2d,
    beta: T,
    c: &'c mut WebBuffer<T>,
    c_shape: Shape2d
}

#[repr(C, packed(4))]
#[derive(Copy, Clone, Debug)]
struct GemmPushConstsV1<T: Num> {
    alpha: T,
    offa: u32,
    rsa: i32,
    csa: i32,
    offb: u32,
    rsb: i32,
    csb: i32,
    beta: T,
    offc: u32,
    rsc: i32,
    csc: i32
}

unsafe impl<T: Num> Zeroable for GemmPushConstsV1<T> {}

unsafe impl<T: Num> Pod for GemmPushConstsV1<T> {}

impl<'a, 'b, 'c, T: Num> Gemm<'a, 'b, 'c, T> {
    pub(super) fn new(a: &'a WebBuffer<T>, b: &'b WebBuffer<T>, c: &'c mut WebBuffer<T>) -> Self {
        Self {
            alpha: T::one(),
            a,
            a_shape: Shape2d::default(),
            b,
            b_shape: Shape2d::default(),
            beta: T::zero(),
            c,
            c_shape: Shape2d::default()
        }
    }
    pub(super) fn a_shape(mut self, a_shape: Shape2d) -> Self {
        self.a_shape = a_shape;
        self
    }
    pub(super) fn b_shape(mut self, b_shape: Shape2d) -> Self {
        self.b_shape = b_shape;
        self
    }
    pub(super) fn c_shape(mut self, c_shape: Shape2d) -> Self {
        self.c_shape = c_shape;
        self
    }
    pub(super) fn exec_v1(&mut self) -> Result<(), Box<dyn Error>> {
        let a_shape = &self.a_shape;
        let [m, k] = a_shape.dim;
        let b_shape = &self.b_shape;
        let [k2, n] = b_shape.dim;
        assert_eq!(k, k2);
        let c_shape = &self.c_shape;
        let [m2, n2] = c_shape.dim;
        assert_eq!(m, m2);
        assert_eq!(n, n2);
        
        let ts = 32;
        
        let m_tile = ts;
        let k_tile = ts;
        let n_tile = ts;
        
        let work_size_x = if m < m_tile {
            1
        }
        else if m % m_tile == 0 {
            m / m_tile
        }
        else {
            m / m_tile + 1
        }; 
        
        let work_size_y = if n < m_tile {
            1
        } 
        else if n % n_tile == 0 {
            n / n_tile
        }
        else {
            n / n_tile + 1
        };
        
        let offa = a_shape.offset;
        let [rsa, csa] = a_shape.strides;
        
        let offb = b_shape.offset;
        let [rsb, csb] = b_shape.strides;
        
        let offc = c_shape.offset;
        let [rsc, csc] = c_shape.strides;
        
        let push_constants = &[GemmPushConstsV1 {
            alpha: self.alpha,
            offa,
            rsa,
            csa,
            offb,
            rsb,
            csb,
            beta: self.beta,
            offc,
            rsc,
            csc
        }];
        
        //dbg!(push_constants);
        
        let push_constants = bytemuck::cast_slice(push_constants);
        
        let src = include_str!("gemm/gemm_v1.comp");
        
        let a = &self.a;
        let gpu = &a.gpu;
        assert_eq!(a.len as u32, m*k);
        let a = &a.buffer;
        
        let b = &self.b;
        assert_eq!(b.len as u32, k*n);
        let b = &b.buffer;
        let mut c = &mut self.c;
        assert_eq!(c.len as u32, m*n);
        let c = &mut c.buffer;
        
        let shader_name = format!(
            "gemm_v1_{}__{}_{}_{}__{}_{}_{}",
            T::type_name(),
            m,
            k,
            n,
            m_tile,
            k_tile,
            n_tile
        );
        
        let device = &gpu.device;
        
        let shader = gpu.shader(&shader_name, || {
            let mut source = String::new();
            source.push_str("#version 450\n");
            source.push_str(&format!("#define T {}\n", T::shader_type()));
            source.push_str(&format!("#define M {}\n", m));
            source.push_str(&format!("#define K {}\n", k));
            source.push_str(&format!("#define N {}\n", n));
            source.push_str(&format!("#define M_TILE {}\n", m_tile));
            source.push_str(&format!("#define K_TILE {}\n", k_tile));
            source.push_str(&format!("#define N_TILE {}\n", n_tile));
            source.push_str(src);
            println!("{}", &source); 
            let spirv = shader_to_spirv(&source, glsl_to_spirv::ShaderType::Compute)?;
            Ok(device.create_shader_module(spirv))
        })?;
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
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
            }],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(a.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(b.slice(..)),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(c.slice(..)),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStage::COMPUTE,
            range: 0..(push_constants.len()*4) as u32
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
            cpass.dispatch(work_size_x as _, work_size_y as _, 1);
        }
        
        gpu.queue.submit(Some(encoder.finish()));
        
        Ok(())
    }
}




