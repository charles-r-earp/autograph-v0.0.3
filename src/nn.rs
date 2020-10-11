use crate::{
    Device, Into2d, Num, RwTensor, Tensor, Tensor2, Tensor4, TensorView, TensorView2, TensorView4,
};
use ndarray::{Dimension, Ix2, Ix4, RemoveAxis};

pub mod builders;
use builders::{Conv2dBuilder, DenseBuilder, MaxPool2dBuilder};

pub mod autograd;
use autograd::{
    Parameter, Parameter1, Parameter2, Parameter4, ParameterD, Variable, Variable2, Variable4,
};

pub mod optimizer;

pub mod saved;

/// Builder struct for 2D Convolution functions\
/// Additional features may be added\
///
/// Defaults:
///   - strides: [1, 1]
///   - padding: [0, 0]
///
/// To use the builder pattern, for example for strides 2 and padding 1
///```
/// let args = Conv2dArgs::default().strides(2).padding(1);
///```
#[derive(Clone, Copy)]
pub struct Conv2dArgs {
    pub(crate) strides: [usize; 2],
    pub(crate) padding: [usize; 2],
}

impl Conv2dArgs {
    /// Sets strides to the given strides (either usize or [usize, usize])
    pub fn strides(mut self, strides: impl Into2d) -> Self {
        self.strides = strides.into_2d();
        self
    }
    // Sets padding to the given padding
    pub fn padding(mut self, padding: impl Into2d) -> Self {
        self.padding = padding.into_2d();
        self
    }
}

impl Default for Conv2dArgs {
    fn default() -> Self {
        Self {
            strides: [1, 1],
            padding: [0, 0],
        }
    }
}

/// Similar to Conv2dArgs, but for pooling functions\
///
/// Defaults:
///   - kernel: [2, 2]
///   - strides: [1, 1]
///   - padding: [0, 0]
#[derive(Clone, Copy)]
pub struct Pool2dArgs {
    pub(crate) kernel: [usize; 2],
    pub(crate) strides: [usize; 2],
    pub(crate) padding: [usize; 2],
}

impl Default for Pool2dArgs {
    fn default() -> Self {
        Self {
            kernel: [2, 2],
            strides: [1, 1],
            padding: [0, 0],
        }
    }
}

impl Pool2dArgs {
    /// Sets the kernel\
    /// Note: unlike Pytorch, does not affect strides
    pub fn kernel(mut self, kernel: impl Into2d) -> Self {
        self.kernel = kernel.into_2d();
        self
    }
    /// Sets the strides
    pub fn strides(mut self, strides: impl Into2d) -> Self {
        self.strides = strides.into_2d();
        self
    }
    /// Sets the padding
    pub fn padding(mut self, padding: impl Into2d) -> Self {
        self.padding = padding.into_2d();
        self
    }
}

/// Trait for Layers\
/// Custom Models should impl Layer
pub trait Layer {
    /// Returns a Vec of all the parameters in the Layer (including its children). Parameter acts like an Arc so it can be cloned to copy references. Layers that do not have parameters (like Activations) do not have to implement this method.
    fn parameters(&self) -> Vec<ParameterD> {
        Vec::new()
    }
    /// Prepares the layer for training if training is true, else prepares for evaluation / inference. This method should be called prior to a forward step ie:
    ///```
    /// for data in training_set {
    ///   let graph = Graph::new();
    ///   let (x, t) = // data
    ///   model.set_training(true);
    ///   let y = model.forward(&x);
    ///   let loss = // loss function
    ///   loss.backward(graph);
    ///   // update model
    /// }
    /// for data in evaluation_set {
    ///   let (x, t) = // data
    ///   model.set_training(false);
    ///   let y = model.forward(&x);
    ///   let loss = // loss function
    /// }
    ///```
    /// The implementation should recursively call set_training on all of its child layers, and or all of its parameters.
    fn set_training(&mut self, training: bool) {}
}

/// Trait for forward pass, implemented by layers\
/// Typically this will call a method or custom Trait method on Variable\
/// A layer like Conv2d will implement Forward, and a model composed of layers will also implement forward.
pub trait Forward<D: Dimension> {
    type OutputDim: Dimension;
    fn forward(&self, input: &Variable<D>) -> Variable<Self::OutputDim>;
}

impl<D: Dimension> Variable<D> {
    /// Convenience method for senquencing several layers:
    ///```
    ///  let y = x.forward(&layer1)
    ///    .forward(&layer2)
    ///    .forward(&layer3)
    ///```
    pub fn forward<D2: Dimension>(&self, layer: &impl Forward<D, OutputDim = D2>) -> Variable<D2> {
        layer.forward(self)
    }
}

/// A linear or fully connected layer with an optional bias
pub struct Dense {
    weight: Parameter2,
    bias: Option<Parameter1>,
}

impl Dense {
    /// Constructs a default DenseBuilder. For example:
    ///```
    /// let dense = Dense::builder()
    ///   .device(&device)
    ///   .inputs(2)
    ///   .outputs(1)
    ///   .bias()
    ///   .build();
    ///```
    /// Device, inputs, and outputs must be specified, there are no defaults. Bias is None if not specified.
    pub fn builder() -> DenseBuilder {
        DenseBuilder::default()
    }
}

impl From<DenseBuilder> for Dense {
    fn from(builder: DenseBuilder) -> Self {
        let device = builder
            .device
            .expect("DenseBuilder requires device to be specified!");
        let inputs = builder
            .inputs
            .expect("DenseBuilder requires inputs to be specified!");
        let outputs = builder
            .outputs
            .expect("DenseBuilder requires outputs to specified!");
        let weight = Parameter::new(RwTensor::zeros(&device, [outputs, inputs]));
        let bias = if builder.use_bias {
            Some(Parameter::new(RwTensor::zeros(&device, outputs)))
        } else {
            None
        };
        Self { weight, bias }
    }
}

impl Layer for Dense {
    fn parameters(&self) -> Vec<ParameterD> {
        let weight = self.weight.clone().into_dyn();
        if let Some(bias) = &self.bias {
            let bias = bias.clone().into_dyn();
            vec![weight, bias]
        } else {
            vec![weight]
        }
    }
    fn set_training(&mut self, training: bool) {
        self.weight.set_training(training);
        if let Some(bias) = &mut self.bias {
            bias.set_training(training);
        }
    }
}

impl Forward<Ix2> for Dense {
    type OutputDim = Ix2;
    fn forward(&self, input: &Variable2) -> Variable2 {
        input.dense(&self.weight, self.bias.as_ref())
    }
}

/// A 2D Convolutional layer
pub struct Conv2d {
    weight: Parameter4,
    bias: Option<Parameter1>,
    args: Conv2dArgs,
}

impl Conv2d {
    /// Constructs a default Conv2dBuilder. For example:
    ///```
    /// let conv2d = Conv2d::builder()
    ///   .device(&device)
    ///   .inputs(2)
    ///   .outputs(1)
    ///   .kernel(3)
    ///   .bias()
    ///   .build();
    ///```
    /// Device, inputs, outputs, and kernel must be specified, there are no defaults. Bias is None if not specified.
    pub fn builder() -> Conv2dBuilder {
        Conv2dBuilder::default()
    }
}

impl From<Conv2dBuilder> for Conv2d {
    fn from(builder: Conv2dBuilder) -> Self {
        let device = builder
            .device
            .expect("Conv2dBuilder requires device to be specified!");
        let inputs = builder
            .inputs
            .expect("Conv2dBuilder requires inputs to be specified!");
        let outputs = builder
            .outputs
            .expect("Conv2dBuilder requires outputs to specified!");
        let [kh, kw] = builder
            .kernel
            .expect("Conv2dBuilder requires kernel to specified!");
        let weight = Parameter::new(RwTensor::zeros(&device, [outputs, inputs, kh, kw]));
        let bias = if builder.use_bias {
            Some(Parameter::new(RwTensor::zeros(&device, outputs)))
        } else {
            None
        };
        Self {
            weight,
            bias,
            args: builder.args,
        }
    }
}

impl Layer for Conv2d {
    fn parameters(&self) -> Vec<ParameterD> {
        let weight = self.weight.clone().into_dyn();
        if let Some(bias) = &self.bias {
            let bias = bias.clone().into_dyn();
            vec![weight, bias]
        } else {
            vec![weight]
        }
    }
    fn set_training(&mut self, training: bool) {
        self.weight.set_training(training);
        if let Some(bias) = &mut self.bias {
            bias.set_training(training);
        }
    }
}

impl Forward<Ix4> for Conv2d {
    type OutputDim = Ix4;
    /// Performs a 2D convolution\
    /// Input: Variable of shape [n, i, ih, iw]\
    ///
    /// Parameters:
    ///   * weight: Tensor of shape [o, i, kh, kw]
    ///   * bias: Optional Tensor of shape [o]
    ///   * args:
    ///     - strides: [sh, sw]
    ///     - padding: [ph, pw]\
    ///
    /// Returns: Variable of shape [n, o, oh, ow]\
    /// where:
    ///  - oh = (ih - kh + 2 * ph) / sh + 1
    ///  - ow = (iw - kw + 2 * pw) / sw + 1
    fn forward(&self, input: &Variable4) -> Variable4 {
        input.conv2d(&self.weight, self.bias.as_ref(), &self.args)
    }
}

/// ReLU Activation\
///
/// see Tensor::relu()
#[derive(Default)]
pub struct Relu {}

impl Layer for Relu {}

impl<D: Dimension> Forward<D> for Relu {
    type OutputDim = D;
    fn forward(&self, input: &Variable<D>) -> Variable<D> {
        input.relu()
    }
}

/// MaxPool2d\
///
/// see Tensor::max_pool2d()
pub struct MaxPool2d {
    args: Pool2dArgs,
}

impl MaxPool2d {
    pub fn builder() -> MaxPool2dBuilder {
        MaxPool2dBuilder::default()
    }
}

impl From<MaxPool2dBuilder> for MaxPool2d {
    fn from(builder: MaxPool2dBuilder) -> Self {
        Self { args: builder.args }
    }
}

impl Layer for MaxPool2d {}

impl Forward<Ix4> for MaxPool2d {
    type OutputDim = Ix4;
    fn forward(&self, input: &Variable4) -> Variable4 {
        input.max_pool2d(&self.args)
    }
}

/// Identity\
///
/// Useful as a placeholder activation
#[derive(Default)]
pub struct Identity {}

impl Layer for Identity {}

impl<D: Dimension> Forward<D> for Identity {
    type OutputDim = D;
    fn forward(&self, input: &Variable<D>) -> Variable<D> {
        input.clone()
    }
}

/// Flatten\
///
/// see Tensor::flatten()
#[derive(Default)]
pub struct Flatten {}

impl Layer for Flatten {}

impl<D: RemoveAxis> Forward<D> for Flatten {
    type OutputDim = Ix2;
    fn forward(&self, input: &Variable<D>) -> Variable2 {
        input.flatten()
    }
}

/// A Sequence of Layers
pub struct Sequential<S>(S);

impl<L> From<Vec<L>> for Sequential<Vec<L>> {
    fn from(layers: Vec<L>) -> Self {
        Sequential(layers)
    }
}

impl<L: Layer> Layer for Sequential<Vec<L>> {
    fn parameters(&self) -> Vec<ParameterD> {
        self.0.iter().flat_map(|layer| layer.parameters()).collect()
    }
    fn set_training(&mut self, training: bool) {
        self.0
            .iter_mut()
            .for_each(|layer| layer.set_training(training));
    }
}

impl<D: Dimension, L: Forward<D, OutputDim = D>> Forward<D> for Sequential<Vec<L>> {
    type OutputDim = D;
    fn forward(&self, input: &Variable<D>) -> Variable<D> {
        self.0
            .iter()
            .fold(input.clone(), |x, layer| layer.forward(&x))
    }
}
