class NeuralNetwork {
  

  constructor(options) {
    // nodes per layer, default is a perceptron
    this.layers = options.layers || [2, 1];
    // how much the weights are changed during training
    this.learningRate = options.learningRate || 0.2; 
    // how the inputs are labeled (optional)
    this.input_labels = options.input_labels || undefined;
    // how the outputs are labeled (optional)
    this.output_labels = options.output_labels || undefined;
    // how many times the NN goes through training in a single training execution 
    this.epochs = options.epochs || 1;
    // splits the training dataset into smaller batch.
    // if this option is undefined, the NN will use the whole dataset without splitting.
    this.batchSize = options.batchSize;
    // number or layers (except the output layer)
    this.n_layers = this.layers.length - 1;

    
    
    // if an array of function is provided, every layer will use the corresponding
    // activation function. if the length of the array doesn't match the number of layers
    // it will warn the user and use the first one.
    // if there are typos or invalid options
    // the functions activation() and d_activation() will use the sigmoid function by default
    
    if(!options.activation) 
      this.activation_f = "sigmoid";
    else {
      // if the property is just a string, it will set the string, 
      // otherwise it sets it to the default function (sigmoid)
      this.activation_f = options.activation;
    }
      
    // checks that the options are compatible together.
    // if something's wrong, it tries to correct it.
    this.checkOptions();

    // the last piece of code assigns random values between -3 and 3 for every weight and bias
    this.weights = [];
    this.bias = [];

    for (let i = 0; i < this.n_layers; i++) {
      // tmp_w will have rows = (number of input nodes) = this.layers[i]
      // and cols = (number of output nodes) = this.layers[i + 1]
    
      let tmp_w = math.random([this.layers[i + 1], this.layers[i]], -3, 3);

      // tmp_b will have rows = (number of output nodes) and
      // cols = 1
      let tmp_b = math.random([this.layers[i + 1]], -3, 3);

      this.weights.push(tmp_w);
      this.bias.push(tmp_b);
    }
    this.lastW = this.weights.length - 1;
  }

  checkOptions(){
    // checks that input labels (if exist) and number of input nodes are the same
    if (this.input_labels && this.input_labels.length !== this.layers[0]){ 
      console.warn("number of input layers and input labels don't match. correcting");
      this.layers[0] = this.input_labels.length;
    }
    // checks that output labels (if exist) and number of output nodes are the same
    if (this.output_labels.length && this.output_labels.length !== this.layers[this.n_layers]){ 
      console.warn("number of output layers and output labels don't match. correcting");
      this.layers[this.n_layers] = this.output_labels.length;
    }
    // if activation_f is an array, checks if the array matches the number of layers.
    if(this.activation_f instanceof Array){
      // if not enough activation functions are not provided only uses the first one
      if (this.activation_f.length < this.n_layers) {
        console.warn("too few activation functions, using only the first one");
        this.activation_f = this.activation_f[0];
      }
      // if there are too many activation functions in the array, it slices it to match
      // the number of layers
      if (this.activation_f.length > this.n_layers){
        console.warn("too many activation functions, deleting the extra ones");
        this.activation_f.slice(0, this.n_layers);
      }
    }
  }

  // methods to set and change various settings
  setActivation(new_activation) {
    if(!options.activation) 
      this.activation_f = "sigmoid";
    else {
      // if the property is just a string, it will set the string, 
      // otherwise it sets it to the default function (sigmoid)
      this.activation_f = new_activation;
    }
    this.checkOptions();
  }

  setLearningRate(lr) {
    this.learningRate = lr;
  }
  setOutputLabels(output_labels) {
    this.output_labels = output_labels;
  }

  setInputLabels(input_labels) {
    this.input_labels = input_labels;
  }

  // does all the prediction calculation and returns an array with all the node values
  // 
  getNodes(input) {
    let nodes = [input];

    for (let i = 0; i < this.n_layers - 1; i++) {
      // weighted sum
      let layer = math.multiply(this.weights[i], nodes[i]);
      // adding biases
      let biased = math.add(this.bias[i], layer);
      // pass through activation function
      if (this.activation_f instanceof Array) {
        nodes.push(activation(biased, this.activation_f[i]));
      } else {
        nodes.push(activation(biased, this.activation_f));
      }
    }
    // weighted sum
    let layer = math.multiply(this.weights[this.lastW], nodes[this.lastW]);

    // adding biases
    let biased = math.add(this.bias[this.lastW], layer);

    // pass through activation function
    nodes.push(biased);
    // returns activated and only weighted node

    return nodes;
  }

  calculate(input){
    // gets last item of getnodes
    let output = this.getNodes(input)[this.n_layers];

    // converts results in probabilities

    // if there are labels
    if (this.output_labels) {
      //gets index of the highest probability
      let index = output.indexOf(Math.max(...output));

      // gets the label of the max probability
      let prediction = this.output_labels[index];

      return prediction;
    }

    return output;
  }

  predict(inputs) {
    let in_nodes = [];
    // if the inputs are labeled creates a vector of numeric inputs
    // corresponding to the relative labels
    if (this.input_labels) {
      for (let input_l of this.input_labels) {
        let inp = inputs[input_l];
        if (inp !== undefined) {
          in_nodes.push(inp);
        } else {
          console.error("input " + input_l + " missing");
        }
      }
    } else {
      //checks if the input is valid
      if (inputs.length != this.layers[0]) {
        console.error("wrong number of inputs");
        return;
      }
      in_nodes = inputs;
    }

    return this.calculate(in_nodes);
  }

  backprops(inputs, target) {
    //checks if the input and target are  valid
    if (inputs.length != this.layers[0]) {
      console.error("wrong number of inputs");
      return;
    } else if (target.length != this.layers[this.n_layers]) {
      console.error("wrong number of targets");
      //return;
    }

    // gets array of the neurons' nodes
    let nodes = this.getNodes(inputs);

    // OUTPUT LAYER //

    // gets output
    let output = nodes[this.n_layers];

    // calculates the output error
    let err = math.subtract(output, target);

    // gets nodes before output nodes
    let lastnodes = nodes[this.n_layers - 1];

    // gets weighted sum of the output nodes
    let z = math.multiply(this.weights[this.lastW], lastnodes);

    // pass through the derivative of the activation function
    let deriv_z;
    if (this.activation_f instanceof Array) {
      deriv_z = d_activation(z, this.activation_f[this.n_layers - 1]);
    } else {
      deriv_z = d_activation(z, this.activation_f);
    }

    // hadamard product between error and the derivative
    let delta = hadamard(err, deriv_z);

    // resizes the delta matrix to match lastnodes size
    delta = math.resize(delta, [math.size(delta)[0], 1]);

    // gradient of the error with respect to the weights
    let gradient = math.multiply(delta, transpose(lastnodes));

    //adjusts the weights matrix
    let variation = math.multiply(this.learningRate, gradient);
    this.weights[this.lastW] = math.subtract(
      this.weights[this.lastW],
      variation
    );

    this.bias[this.lastW] = math.subtract(
      this.bias[this.lastW],
      math.multiply(this.learningRate, math.squeeze(delta))
    );

    // this.n_layer = this.lastW + 1
    for (let i = 1; i < this.n_layers; i++) {
      // gets nodes before current nodes
      lastnodes = nodes[this.lastW - i];

      // gets precedent weights matrix
      let lastweights = math.transpose(this.weights[this.lastW - i + 1]);
      err = math.multiply(lastweights, delta);

      // gets weighted sum of the current nodes
      z = math.multiply(this.weights[this.lastW - i], lastnodes);
      // pass through the derivative of the activation function

      let deriv_z;
      if (this.activation_f instanceof Array) {
        deriv_z = d_activation(z, this.activation_f[this.lastW - i]);
      } else {
        deriv_z = d_activation(z, this.activation_f);
      }

      // calculates current delta
      delta = hadamard(err, deriv_z);

      // resizes the delta matrix to match lastnodes size
      delta = math.resize(delta, [delta.length, 1]);

      // gradient of the error with respect to the weights
      gradient = math.multiply(delta, transpose(lastnodes));

      // calculates the variation of the matrix
      variation = math.multiply(this.learningRate, gradient);

      //adjusts the weights and bias matrix
      this.weights[this.lastW - i] = math.subtract(
        this.weights[this.lastW - i],
        variation
      );
      this.bias[this.lastW - i] = math.subtract(
        this.bias[this.lastW - i],
        math.multiply(this.learningRate, math.resize(delta, [delta.length]))
      );
    }
  }

  train(dataset) {
    //shuffles the dataset
    dataset = data_shuffle(dataset);
    let lasti = 0;
    let size = this.batchSize || dataset.length;

    let target_arr = [];
    // divides the datasets in batches
    for (let iter = 0; iter < this.epochs; iter++) {
      for (let i = lasti; i < lasti + size && lasti < dataset.length; i++) {
        // if the inputs are labeled creates a vector of numeric inputs
        // corresponding to the relative labels
        let inputs = [];
        if (this.input_labels) {
          for (let input_l of this.input_labels) {
            let inp = dataset[i][input_l];
            if (inp !== undefined) {
              inputs.push(inp);
            } else {
              console.error("input " + input_l + " missing");
            }
          }
        } else {
          inputs = dataset[i].inputs;
          //checks if the input is valid
          if (inputs.length != this.layers[0]) {
            console.error("wrong number of inputs");
            return;
          }
        }
        // creates a vector with the correct answers
        // otherwise gives the unlabeled targets from the dataset
        if (this.output_labels) {
          target_arr = [];
          // loops through the labels, if the target matches the label
          // then the probability is 1, otherwise is 0
          for (let j = 0; j < this.output_labels.length; j++) {
            if (dataset[i].target == this.output_labels[j]) {
              target_arr.push(1);
            } else {
              target_arr.push(0);
            }
          }
          nn.backprops(inputs, target_arr);
        } else {
          nn.backprops(inputs, dataset[i].target);
        }
      }
    }
  }

  test(dataset) {
    // creates error variables all set to 0
    let error = 0;
    let errSS = math.zeros(dataset[0].target.length);
    let totSS = math.zeros(dataset[0].target.length);
    let mean;

    if (!this.output_labels) {
      mean = getAverage(dataset);
    }
    for (let sample of dataset) {
      let guess;
      if (this.input_labels !== undefined) {
        guess = this.predict(sample);
      } else {
        guess = this.predict(sample.inputs);
      }
      // checks if this.labels is defined
      if (this.output_labels) {
        // just encreases the number of wrong guesses
        if (guess != sample.target) error++;
      } else {
        // calculates the parameters for the R^2 formula
        totSS = math.add(totSS, TSS(sample.target, mean));
        errSS = math.add(errSS, ESS(sample.target, guess));
      }
    }

    if (this.output_labels) {
      // calculates the percentage of correct guesses
      return (1 - error / dataset.length) * 100;
    } else {
      // calculates R^2 = 1-(ESS/TSS)
      let ratio = math.divide(errSS, totSS);
      return math.subtract(1, ratio);
    }
  }

  saveState() {
    // returns an object with all the NN information
    let state = {
      weights: this.weights,
      biases: this.bias,
      learningRate: this.learningRate,
      activation: this.activation_f,
      inputs: this.input_labels,
      outputs: this.output_labels,
      epochs: this.epochs,
      layers: this.layers
    };
    return state;
  }

  loadState(state) {
    // loads the information from a state object
    this.weights = state.weights;
    this.bias = state.biases;
    this.learningRate = state.learningRate;
    this.activation_f = state.activation;
    this.input_labels = state.inputs || undefined;
    this.output_labels = state.outputs || undefined;
    this.epoch = state.epochs;
    this.layers = state.layers;
  }

  reset() {
    // resets everything with random weights and biases
    this.weights = [];
    this.bias = [];

    for (let i = 0; i < this.n_layers; i++) {
      // tmp_w will have rows = (number of input nodes) and
      // cols = (number of output nodes)
      let tmp_w = math.random([this.layers[i + 1], this.layers[i]], -3, 3);

      // tmp_b will have rows = (number of output nodes) and
      // cols = 1
      let tmp_b = math.random([this.layers[i + 1]], -3, 3);

      this.weights.push(tmp_w);
      this.bias.push(tmp_b);
    }
  }
}
