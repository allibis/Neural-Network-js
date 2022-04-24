# Neural-Network-js
 Neural Network written in JavaScript with the help of [Daniel Shiffman's videos](https://youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh)

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

- `checkOptions`: checks that every properties is consistent (and doesn't generate errors).

## Properties

- `layers`: it holds the structure of the neural network. it's an array of number, each one is the number of nodes in every layer. the default value is [2, 1] (the Perceptron).

- `learningRate`: it's how much every weight and bias is changed during backpropagation.

- `input_labels`: (optional) the labels for the input. if they are specified, they must be specified too in the dataset as keys for every element in the dataset, while the values are numbers.

- `output_labels`: (optional) the labels for the output. if specified, they are the possible results for every prediction.

- `activation`: it's a string or an array of string. in the second case, every element in the array is used for the corresponding layer. the functions implemented are:
    1. sigmoid
    2. relu (Rectified linear units)
    3. softplus
    4. gaussian
    5. sine

- `epochs`: how many times the whole dataset is passed through.

- `batchSize`: how many elements of dataset are in every batch.

- `n_layers`: number of layers-1 (it's for array purpose, since it's the last index of the arrays).

- `weights`: contains all weights. the object is an array of arrays. every sub array corresponds to each layer. 

- `bias`: it's an array, every element is the corresponding layer's bias. 

- `lastW`: the index of the last weight array in the `weights` array.

## examples

