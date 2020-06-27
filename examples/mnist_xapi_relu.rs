use autograph::{
  Device, Cpu,
  DataRef,
  TensorBase,
  Tensor, 
  TensorView,
  TensorViewMut,
  ArcTensor,
  Pool2dArgs
};
use autograph::autograd::{
  Graph,
  Variable, Variable2, Variable4,
  ParameterD
};
use autograph::layer::{
  Layer, Forward, 
  Conv2d, Dense, MaxPool2d
};
use autograph::utils::classification_accuracy;
use autograph::datasets::Mnist; // requires feature datasets
use std::time::Instant;
use ndarray::{Dimension, Ix2, Ix4};
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::Normal;
use num_traits::ToPrimitive;
use argparse::{ArgumentParser, Store};

// This example shows how to add new ops to autograph. 
// Requires features datasets and xapi 
// We will add a custom Relu Layer, and use it to train a model

// Computes the output of relu on cpu
// Like ArrayView, TensorView represents borrowed data
fn my_relu_cpu<D: Dimension>(input: &TensorView<f32, D>, output: &mut TensorViewMut<f32, D>) {
  // xapi_as_mut_cpu_slice() returns Some if the data is on the cpu
  output.xapi_as_mut_cpu_slice()
    .unwrap()
    .iter_mut()
    .zip(input.xapi_as_cpu_slice().unwrap())
    .for_each(|(y, &x)| {
      if x >= 0. {
        *y = x;
      }
      else {
        *y = 0.;
      }
    });
}

// This function computes the input gradient, and is generic over the dimension of the tensors
fn my_relu_backward_cpu<D: Dimension>(input: &TensorView<f32, D>, input_grad: &mut TensorViewMut<f32, D>, output_grad: &TensorView<f32, D>) {
  // This implementation will use pure Rust and is sequential. In autograph, oneDNN is used for cpu ops which has a parallel implementation 
  // Methods prefixed by xapi are enabled with feature xapi
  // This is experimental and subject to change
  input.xapi_as_cpu_slice()
    .unwrap() // panics if the data is not on the cpu
    .iter()
    .zip(input_grad.xapi_as_mut_cpu_slice().unwrap())
    .zip(output_grad.xapi_as_cpu_slice().unwrap())
    .for_each(|((&x, dx), &dy)| {
      if x > 0. {
        *dx += dy; 
      }
    });
}

// This function potentially selects an implementation based on device
fn my_relu_backward<D: Dimension>(input: &TensorView<f32, D>, input_grad: &mut TensorViewMut<f32, D>, output_grad: &TensorView<f32, D>) {
  match input.device() {
    Device::Cpu(_) => my_relu_backward_cpu(input, input_grad, output_grad),
    #[allow(unreachable_patterns)]
    _ => unimplemented!() 
  }
}

// traits are used to add methods to types
trait MyReluExt {
  type Output;
  fn my_relu(&self) -> Self::Output;
}

impl<S: DataRef<Elem=f32>, D: Dimension> MyReluExt for TensorBase<S, D> {
  type Output = Tensor<f32, D>; // By convention, tensor ops return a Tensor
  fn my_relu(&self) -> Tensor<f32, D> {
    // Allocate data on the device
    let mut output = unsafe { Tensor::xapi_uninitialized(self.device(), self.raw_dim()) };
    match self.device() {
      Device::Cpu(_) => my_relu_cpu(&self.view(), &mut output.view_mut()),
      #[allow(unreachable_patterns)]
      _ => unimplemented!() // Here we would add a cuda implementation 
    }
    output 
  }
} 

// Implementation for MyReluExt for Variable 
impl<D: Dimension> MyReluExt for Variable<D> {
  type Output = Self;
  fn my_relu(&self) -> Self {
    // Get the graph from the input
    // if no graph was provided or the graph has been dropped, graph will be None
    let graph = self.xapi_graph();
    // Construct a new output with our TensorBase::my_relu() method
    let output = Self::new(
      graph.as_ref(),
      self.value().my_relu(),
      self.grad().is_some() // The output requires grad if any inputs require grad
    );
    if let Some(output_grad) = output.grad() { // If we have an output gradient
      let graph = graph.unwrap(); // We can assume that we have a graph
      // Prepare tensors for the backward op
      // We have to clone them so that they can be moved into the closure
      // The data will be dropped appropriately, since it is wrapped in an Arc
      // Moving into IxDyn means that the tensors have an explicit dimension, which is 'static
      // Custom backward ops are moved into a Box<dyn Fn()> which requires 'static
      let output_grad = output_grad.clone()
        .xapi_into_dyn(); 
      let input = self.value().clone()
        .into_dyn();
      let input_grad = self.grad()
        .unwrap() // Here we can assume that if we have an output grad we have an input grad
        .clone()
        .xapi_into_dyn();
      graph.xapi_backward_variable_op(move || {
        // Lock the input_grad for exclusive write access
        // The lock will be released at the end of the closure
        let mut input_grad = input_grad.write()
          .unwrap(); // Unwrap the LockResult
        // Lock the output grad for shared read access
        // If the output_grad has not been written to, read() will return None
        let output_grad = output_grad.read()
          .unwrap() // Unwrap the Option
          .unwrap(); // Unwrap the LockResult
        // Compute the backward operation
        my_relu_backward(&input.view(), &mut input_grad.view_mut(), &output_grad.view()); 
      });
    }
    output
  }
}
  
#[derive(Default)]
struct MyRelu {}

impl Layer for MyRelu {}

impl<D: Dimension> Forward<D> for MyRelu {
  type OutputDim = D;
  fn forward(&self, input: &Variable<D>) -> Variable<D> {
    input.my_relu()
  }
}

pub struct Net {
  conv1: Conv2d,
  relu1: MyRelu,
  pool1: MaxPool2d,
  dense1: Dense 
}

impl Net {
  fn new(device: &Device) -> Self {
    let conv1 = Conv2d::builder()
      .device(device)
      .inputs(1)
      .outputs(8)
      .kernel(7)
      .build();
    let relu1 = MyRelu::default();
    let pool1 = MaxPool2d::builder()
      .args(
        Pool2dArgs::default()
          .kernel(2)
          .strides(2)
      )
      .build();
    let dense1 = Dense::builder()
      .device(&device)
      .inputs(8*11*11)
      .outputs(10)
      .bias()
      .build();
    Self {
      conv1,
      relu1,
      pool1,
      dense1
    }
  }
}

impl Layer for Net {
  fn parameters(&self) -> Vec<ParameterD> {
    self.conv1.parameters()
      .into_iter()
      .chain(self.dense1.parameters())
      .collect()
  }
  fn set_training(&mut self, training: bool) {
    self.conv1.set_training(training);
    self.dense1.set_training(training);
  }
}

impl Forward<Ix4> for Net {
  type OutputDim = Ix2;
  fn forward(&self, input: &Variable4) -> Variable2 {
    input.forward(&self.conv1)
      .forward(&self.relu1)
      .forward(&self.pool1)
      .flatten()
      .forward(&self.dense1)
  }
}

fn main() {
  // Use argparse to get command line arguments
  let (epochs, lr, train_batch_size, eval_batch_size) = {
    let mut epochs = 10;
    let mut lr = 0.001;
    let mut train_batch_size: usize = 100;
    let mut eval_batch_size: usize = 1000;
    {
      let mut ap = ArgumentParser::new();
      ap.set_description("MNIST Extend API ReLU Example");
      ap.refer(&mut epochs)
        .add_option(&["-e", "--epochs"], Store, "Number of epochs to train for.");
      ap.refer(&mut lr)
        .add_option(&["--learning-rate"], Store, "Learning Rate");
      ap.refer(&mut train_batch_size)
        .add_option(&["--train-batch_size"], Store, "Training Batch Size");
      ap.refer(&mut eval_batch_size)
        .add_option(&["--eval-batch-size"], Store, "Evaluation Batch Size");
      ap.parse_args_or_exit();
    }
    (epochs, lr, train_batch_size, eval_batch_size)
  };
  
  // Our relu layer will only be implemented for cpu 
  let device = Device::from(Cpu::new());
  
  println!("epochs: {}", epochs);
  println!("lr: {}", lr);
  println!("train_batch_size: {}", train_batch_size);
  println!("eval_batch_size: {}", eval_batch_size);
  println!("device: {:?}", &device);
  
  let mut rng = SmallRng::seed_from_u64(0); 
  
  let mut model = Net::new(&device);
  
  model.parameters()
    .into_iter()
    .for_each(|w| {
      let dim = w.value().raw_dim();
      if dim.ndim() > 1 { // Leave biases as zeros
        w.value()
          .write()
          .unwrap()
          .fill_random(&Normal::new(0., 0.01).unwrap(), &mut rng)
      }
    });
  
  // Create a Mnist dataset
  // See http://yann.lecun.com/exdb/mnist/
  // Loads the dataset into memory, potentially downloading and saving it to datasets/mnist/*
  let dataset = Mnist::new();

  // Record the time 
  let start = Instant::now();
  
  // Iterate for epochs
  for epoch in 1 ..= epochs {
    let mut train_loss = 0.;
    let mut train_correct: usize = 0;
    // Training loop: iterate over the training set
    // Mnist::train() returns an iterator of (images: ArrayView4<u8>, labels: ArrayView1<u8>) 
    dataset.train(train_batch_size)
      .for_each(|(x_arr, t_arr)| {
        // Set the parameters to training mode
        // Note that this also releases any gradients from a previous backward pass
        model.set_training(true);
        // Construct a graph. A graph is used to create Variables, and to enqueue operations that will be used to compute gradients
        let graph = Graph::new(); // This is an Arc<Graph> so that it can be shared
        // Construct a Variable with the graph and data from the dataset 
        let x = Variable::new(
          Some(&graph),
          // Construct a tensor from the input array view
          // Tensor::to_f32() converts the u8 data to f32 on the device
          Tensor::from_array(&device, x_arr.view())
            .to_f32(),
          // requires_grad is false because this is an input, we don't need to compute it's gradient 
          false
        );
        // Construct an ArcTensor for our target
        // t_arr is a batch of input labels that are converted to a matrix of shape [batch_size, nclasses=10]
        // ArcTensor::from(Tensor) consumes (ie moves) the tensor, the data is not copied
        // t must be an ArcTensor because it will be stored in the graph to compute the gradient of the model prediction y
        let t = ArcTensor::from(
          Tensor::from_array(&device, t_arr.view())
            .to_one_hot_f32(10)
        );
        // Operations on Variable enqueue backward ops into the graph, if provided, to compute gradients
        let y = model.forward(&x);
        // Compute the total loss of our model prediction. Cross entropy is used for classification. 
        let loss = y.cross_entropy_loss(&t);
        // Compute the backward pass
        // Here graph is consumed (ie dropped). Variables x, y, and loss will no longer be tied to this graph and any further operations will not compute gradients. 
        // The gradient of y will be computed, then the gradients of w and b
        loss.backward(graph);
        model.parameters()
          .into_iter()
          .for_each(|w| {
            // Lock the weight value, acquiring exclusive write access
            let mut w_value = w.value()
              .write()
              .unwrap(); // Unwraps the LockResult
            // Lock the weight gradient, acquiring shared read access
            let w_grad = w.grad()
              .unwrap()
              .read()
              .unwrap() // Unwraps the Option, would be None if backward was not called
              .unwrap(); // Unwraps the LockResult
            // Update the weight
            w_value.scaled_add(-lr, &w_grad);
          });
        train_correct += classification_accuracy(&y.value().as_array().view(), &t_arr);
        train_loss += loss.value().as_slice()[0];
      });
    train_loss /= 60_000f32;
    let train_acc = train_correct.to_f32().unwrap() * 100f32 / 60_000f32; 
    
    let mut eval_loss = 0.;
    let mut eval_correct: usize = 0;
    dataset.eval(eval_batch_size)
      .for_each(|(x_arr, t_arr)| {
        // Here we set training to false, which prevents gradients from being computed
        model.set_training(false);
        // Construct a variable, without a graph
        let x = Variable::new(
          None,
          Tensor::from_array(&device, x_arr.view())
            .to_f32(),
          false
        );
        let t = ArcTensor::from(
          Tensor::from_array(&device, t_arr.view())
            .to_one_hot_f32(10)
        );
        // Perform the same operation as before, but only execute the forward pass (ie inference)
        let y = model.forward(&x);
        let loss = y.cross_entropy_loss(&t);
        eval_correct += classification_accuracy(&y.value().as_array().view(), &t_arr);
        eval_loss += loss.value().as_slice()[0];
      });
    eval_loss /= 10_000f32;
    let eval_acc = eval_correct.to_f32().unwrap() * 100f32 / 10_000f32;
    let elapsed = Instant::now() - start;
    println!("epoch: {} elapsed {:.0?} train_loss: {:.5} train_acc: {:.2}% eval_loss: {:.5} eval_acc: {:.2}%", 
      epoch, elapsed, train_loss, train_acc, eval_loss, eval_acc);
  }
}
