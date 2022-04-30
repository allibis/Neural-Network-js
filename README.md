# Neural-Network-js
 Neural Network written in JavaScript 

## Methods
- `predict(inputs)`: returns either a number or a string, corresponding to the prediction made from the inputs. the `input` is an array of numbers matching the number of nodes in the first layer.

- `train(dataset)`: trains the neural network using the `dataset` provided. changing the value of `learningRate`, `batchSize` and `epochs` will change the time needed to train the network.

- `test(dataset)`: runs the neural network using the `dataset` and calculates the R² error if the outputs are numerical. otherwise it returns the percentage of correct guesses.

- `saveState()`: returns an object that holds all the NN properties. 

- `loadState(state)`: loads a state object returned by the `saveState()` function.

- `getNodes(inputs)`: calculate the node values based on a given input. returns an array of arrays, each one corresponding to a layer. the last array is the raw result of the prediction.

- `backprops(inputs, target)`: given inputs and targets, it backpropagates the error changing the weights and biases.

- `setLearningRate(learningRate)`: sets new learning rate value.

- `setActivation(new_activation)`: sets a new activation function or an array of activation functions, each one corresponding to a layer.

- `setInputLabels(output_labels)`: sets new input label(s).

- `setOutputLabels(output_labels)`: sets new output label(s).

- `reset()`: resets the weights and biases.

- `normalize(dataset)`: normalizes the dataset scaling all inputs in a range between 0 and 1.

## Properties

- `layers`: it holds the structure of the neural network. it's an array of number, each one is the number of nodes in every layer. the default value is [2, 1] (the Perceptron).

- `learningRate`: it's how much every weight and bias is changed during backpropagation.

- `input_labels`: (optional) the labels for the input. if they are specified, they must be specified too in the dataset as keys for every element in the dataset, while the values are numbers.

- `output_labels`: (optional) the labels for the output. if specified, they are the possible results for every prediction. in that case the output will be only one and in the dataset only one target string can be provided.

- `activation`: it's a string or an array of string. in the second case, every element in the array is used for the corresponding layer. the functions implemented are:
    1. sigmoid
    2. relu (Rectified linear units)
    3. softplus
    4. gaussian
    5. sine
    6. softmax

- `epochs`: how many times the whole dataset is passed through.

- `batchSize`: how many elements of dataset are in every batch.

- `n_layers`: number of layers-1 (it's for array purpose, since it's the last index of the arrays).

- `weights`: contains all weights. the object is an array of arrays. every sub array corresponds to each layer. 

- `bias`: it's an array, every element is the corresponding layer's bias. 

- `lastW`: the index of the last weight array in the `weights` array.

- `min_input` and `max_input`: numbers used to normalize data. the formula used is (input - min_input)/(max_input - min_input).

- `normalize_data`: flag that allows the NN to automatically normalize the data.

## option object

the options object is a js object with this structure:

```
const options = 
{
    layers: Array, // array of integers
    learningRate: Number,
    input_labels: Array, // array of strings, optional
    input_labels: Array, // array of strings, optional
    epochs: Integer,
    batchSize: Integer,
    activation: String || Array, // array of strings, optional, default is ReLU
    min_input: Number, // smaller input provided, optional
    max_input: Number, // bigger input provided, optional
    normalize_data: Bool // if true, the NN will automatically normalize the dataset
}
```

## examples

if the option object looks like this

```
const options = {
  layers: [2, 4, 2],
  learningRate: 0.5,
  output_labels: ["true", "false"],
  epochs: 200
};
```

a corresponding dataset element could be:

```
{
    inputs: [1, 1]
    target: ["false"]
}
```

in this case only one target can be set, so this element would not work:

```
{
    ...
    target: ["true", "false"]
}
```

if `output_labels` is unspecified, then the element would be:

```
{
    inputs: [1, 1],
    target: [0]
}
```

if `input_labels` is specified, then the dataset element must have a key for every input label, each one holding a number: 

```
const options = {
    ...
    input_labels: ["a", "b", ... , "z"];
    ...
}
```

dataset: 

```
{
    a: val_a,
    b: val_b,
    ...
    z: val_z,
    ...
}
```
## Resources
- [Backpropagation](https://towardsdatascience.com/a-10-line-proof-of-back-propagation-5a2cad1032c4)
- [Daniel Shiffman's Nature of code](https://youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh)
- [Machine Learning](https://developers.google.com/machine-learning)
- [3B1B Neural Network](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
