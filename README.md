[![License-MIT/Apache-2.0](https://img.shields.io/badge/license-MIT/Apache_2.0-blue.svg)](https://github.com/charles-r-earp/autograph/blob/master/LICENSE-APACHE)
![Rust](https://github.com/charles-r-earp/autograph/workflows/Rust/badge.svg?branch=master)
# autograph
Machine Learning Library for Rust

# Features
  - Safe API
  - Thread Safe
  - CPU and CUDA are fully supported
  - Flexible (Dynamic Backward Graph)

## Layers
  - Dense
  - Conv2d
  - MaxPool2d
  - Relu
 
## Loss Functions
  - CrossEntropyLoss
 
## Datasets
  - MNIST

# Graphs
During training, first a forward pass is run to determine the outputs of a model and compute the loss. Then a backward pass is run to compute the gradients which are used to update the model parameters. Autograph constructs a graph of operations in the forward pass that is then used for the backward pass. This allows for intermediate values and gradients to be lazily allocated and deallocated using RAII in order to minimize memory usage. Native control flow like loops, if statements etc. can be used to define a forward pass, which does not need to be the same each time. This allows for novel deep learning structures like RNN's and GAN's to be constructed, without special hardcoding. 

# Supported Platforms
Tested on Ubuntu-18.04, Windows Server 2019, and macOS Catalina 10.15. Generally you will want to make sure that OpenMP is installed. Currently cmake / oneDNN has trouble finding OpenMP on mac and builds in sequential mode. This greatly degrades performance (approx 10x slower) but it will otherwise run without issues. If you have trouble building autograph, please create an issue. 

# CUDA
Cuda can be enabled by passing the feature "cuda" to cargo. CUDA https://developer.nvidia.com/cuda-downloads and cuDNN https://developer.nvidia.com/cudnn must be installed. See https://github.com/bheisler/RustaCUDA and https://github.com/charles-r-earp/cuda-cudnn-sys for additional information. 

# Datasets
Autograph includes a datasets module enabled with the features datasets. This currently has the MNIST dataset, which is downloaded and saved automatically. The implementation of this is old and outdated (it uses reqwest among others which now uses async), and compiles slowly. Potentially overkill for such a small dataset, but for adding new datasets (like ImageNet), we will need an updated, fast implementation. 

# Getting Started
If you are new to Rust, you can get it and find documentation here: https://www.rust-lang.org/

If you have git installed (see https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) you can clone the repo by:
```
git clone https://github.com/charles-r-earp/autograph
```

OpenMP is used when available in oneDNN. Without it, execution will be very slow. This requires a C++ compiler that supports OpenMP 2.0 or later:
  - gcc 4.2.0 or later `gcc --version`
  - clang/llvm 
    - Linux with Clang: may need to install `libomp-dev` for Debian/Ubuntu, `libomp-devel` for Void

View the documentation by:
```
cargo doc --open [--features "[datasets] [cuda]"]
```
To add autograph to your project add it as a dependency in your cargo.toml (features are optional):
```
[dependencies]
autograph = { version = 0.0.2, features = ["datasets", "cuda"] }
// or from github
autograph = { git = https://github.com/charles-r-earp/autograph, features = ["datasets", "cuda"] }
```

# Tests
Run the unit-tests with (passing the feature cuda additionally runs cuda tests):
```
cargo test --lib [--features cuda]
```

# Examples
Run the examples with:
```
cargo run --example [example] --features "datasets [cuda]" --release
```
See the examples directory for the examples.

# Benchmarks
Run the benchmarks with:
```
cargo bench [--features cuda]
```

# Roadmap 
  - Optimizers (SGD, Adam)
  - Saving and loading of models / Serde
  - Data transfers between devices (local model parallel)
  - Data Parallel (multi-gpu)
  - Remote Distributed Parallel (ie training on multiple machines)
