use autograph::Cpu;
#[cfg(feature = "rocm")]
use autograph::RocmGpu;

#[test]
fn test_new_cpu() {
    Cpu::new();
}

#[cfg(feature = "rocm")]
#[test]
fn test_new_rocm_gpu() {
    RocmGpu::new(0);
}
