#![allow(warnings)]
use argparse::{ArgumentParser, Store, StoreTrue};
use autograph::autograd::{Graph, ParameterD, Variable, Variable2, Variable4};
use autograph::datasets::Mnist; // requires feature "datasets"
use autograph::layer::{Conv2d, Dense, Forward, Layer};
use autograph::utils::classification_accuracy;
use autograph::{ArcTensor, Device, Pool2dArgs, Tensor, Tensor2, Tensor4, TensorView4};
use ndarray::{Dimension, Ix2, Ix4};
use num_traits::ToPrimitive;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

// A version of the LeNet5 Model
struct Lenet5 {
    conv1: Conv2d,
    conv2: Conv2d,
    dense1: Dense,
    dense2: Dense,
    dense3: Dense,
}

impl Lenet5 {
    // new is the primary constructor for a struct
    // Here we construct the model on the given device
    // Note that currently Conv2d and Dense layers fill their parameters with zeros, so the model must be manually initialized
    pub fn new(device: &Device) -> Self {
        let conv1 = Conv2d::builder()
            .device(&device)
            .inputs(1)
            .outputs(6)
            .kernel(5)
            .build();
        let conv2 = Conv2d::builder()
            .device(&device)
            .inputs(6)
            .outputs(16)
            .kernel(5)
            .build();
        let dense1 = Dense::builder()
            .device(&device)
            .inputs(256)
            .outputs(120)
            .build();
        let dense2 = Dense::builder()
            .device(&device)
            .inputs(120)
            .outputs(84)
            .build();
        let dense3 = Dense::builder()
            .device(&device)
            .inputs(84)
            .outputs(10)
            .bias()
            .build();
        Self {
            conv1,
            conv2,
            dense1,
            dense2,
            dense3,
        }
    }
}

// Layer is a core trait for Layers and Models
impl Layer for Lenet5 {
    // Gathers all the parameters in the model
    fn parameters(&self) -> Vec<ParameterD> {
        self.conv1
            .parameters()
            .into_iter()
            .chain(self.conv2.parameters())
            .chain(self.dense1.parameters())
            .chain(self.dense2.parameters())
            .chain(self.dense3.parameters())
            .collect()
    }
    // Prepares the model for training (or evaluation)
    fn set_training(&mut self, training: bool) {
        self.conv1.set_training(training);
        self.conv2.set_training(training);
        self.dense1.set_training(training);
        self.dense2.set_training(training);
        self.dense3.set_training(training);
    }
}

// Forward is a trait for Layers and Models
// Forward executes the forward pass, returning the prediction of the model
impl Forward<Ix4> for Lenet5 {
    type OutputDim = Ix2;
    fn forward(&self, input: &Variable4) -> Variable2 {
        let pool_args = Pool2dArgs::default().kernel(2).strides(2);
        // Variable has a forward(layer: impl Forward) method
        // This makes it easy to chain several operations
        input
            .forward(&self.conv1)
            .relu()
            .max_pool2d(&pool_args)
            .forward(&self.conv2)
            .relu()
            .max_pool2d(&pool_args)
            .flatten()
            .forward(&self.dense1)
            .relu()
            .forward(&self.dense2)
            .relu()
            .forward(&self.dense3)
    }
}

fn main() {
    let (epochs, lr, train_batch_size, eval_batch_size) = {
        let mut epochs = 50;
        let mut lr = 0.001;
        let mut train_batch_size: usize = 256;
        let mut eval_batch_size: usize = 512; // bugged, must use smaller batch size 1024;
        {
            let mut ap = ArgumentParser::new();
            ap.set_description("MNIST Lenet5 Example");
            ap.refer(&mut epochs).add_option(
                &["-e", "--epochs"],
                Store,
                "Number of epochs to train for.",
            );
            ap.refer(&mut lr)
                .add_option(&["--learning-rate"], Store, "Learning Rate");
            ap.refer(&mut train_batch_size).add_option(
                &["--train-batch-size"],
                Store,
                "Training Batch Size",
            );
            ap.refer(&mut eval_batch_size).add_option(
                &["--eval-batch-size"],
                Store,
                "Evaluation Batch Size",
            );
            ap.parse_args_or_exit();
        }
        (epochs, lr, train_batch_size, eval_batch_size)
    };

    let device = Device::default();

    println!("epochs: {}", epochs);
    println!("lr: {}", lr);
    println!("train_batch_size: {}", train_batch_size);
    println!("eval_batch_size: {}", eval_batch_size);
    println!("device: {:?}", &device);

    let mut rng = SmallRng::seed_from_u64(0);

    let mut model = Lenet5::new(&device);
    model.parameters().into_iter().for_each(|w| {
        let dim = w.value().raw_dim();
        if dim.ndim() > 1 {
            // Leave biases as zeros
            w.value()
                .write()
                .unwrap()
                .fill_random(&Normal::new(0., 0.01).unwrap(), &mut rng)
        }
    });

    let dataset = Mnist::new();

    let start = Instant::now();
    for epoch in 1..=epochs {
        let mut train_loss = 0.;
        let mut train_correct: usize = 0;
        dataset.train(train_batch_size).for_each(|(x_arr, t_arr)| {
            model.set_training(true);
            let graph = Graph::new();
            let x = Variable::new(
                Some(&graph),
                Tensor::from_array(&device, x_arr).to_f32(),
                false,
            );
            let t = ArcTensor::from(Tensor::from_array(&device, t_arr).to_one_hot_f32(10));
            let y = model.forward(&x);
            let loss = y.cross_entropy_loss(&t);
            loss.backward(graph);
            //println!("y: {:?} loss: {:?}", y.value().as_slice(), loss.value().as_slice());
            model.parameters().iter().for_each(|w| {
                let mut w_value = w.value().write().unwrap();
                let w_grad = w.grad().unwrap().read().unwrap().unwrap();
                w_value.scaled_add(-lr, &w_grad);
            });
            train_correct += classification_accuracy(&y.value().as_array().view(), &t_arr);
            train_loss += loss.value().as_slice()[0];
        });
        //panic!();
        train_loss /= 60_000f32;
        let train_acc = train_correct.to_f32().unwrap() * 100f32 / 60_000f32;

        let mut eval_loss = 0.;
        let mut eval_correct: usize = 0;
        dataset.eval(eval_batch_size).for_each(|(x_arr, t_arr)| {
            model.set_training(false);
            let x = Variable::new(None, Tensor::from_array(&device, x_arr).to_f32(), false);
            let t = ArcTensor::from(Tensor::from_array(&device, t_arr).to_one_hot_f32(10));
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
