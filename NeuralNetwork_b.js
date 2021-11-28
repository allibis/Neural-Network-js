class NeuralNetwork {
  constructor(options) {
    this.layers = options.layers || [2, 1];
    this.learningRate = options.learningRate || 0.2;
    this.inputs = options.inputs;
    this.labels = options.labels;
    this.epochs = options.epochs || 1;

    this.n_layers = this.layers.length - 1;
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
    this.lastW = this.weights.length - 1;
  }

  getNodes(input) {
    let nodes = [input];

    for (let i = 0; i < this.n_layers; i++) {
      // weighted sum
      let layer = math.multiply(this.weights[i], nodes[i]);

      // adding biases
      let biased = math.add(this.bias[i], layer);

      // pass through activation function
      nodes.push(activation(biased));
    }
    // returns activated and only weighted node
    return nodes;
  }

  predict(inputs) {
    let nodes = inputs;
    //checks if the input is valid
    if (inputs.length != this.layers[0]) {
      console.error("wrong number of inputs");
      return;
    }

    // gets last item of getnodes
    let output = this.getNodes(inputs)[this.n_layers];

    // converts results in probabilities

    // if there are labels
    if (this.labels) {
      //gets index of the highest probability
      let index = output.indexOf(Math.max(...output));

      // gets the label of the max probability
      let prediction = this.labels[index];

      return prediction;
    }
    return output;
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

    //gets output
    let output = nodes[this.n_layers];

    // calculates the output error
    let err = math.subtract(output, target);

    // gets nodes before output nodes
    let lastnodes = nodes[this.n_layers - 1];

    // gets weighted sum of the output nodes
    let z = math.multiply(this.weights[this.lastW], lastnodes);

    // pass through the derivative of the activation function
    let deriv_z = d_activation(z);

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
      nn;

      // gets weighted sum of the current nodes
      z = math.multiply(this.weights[this.lastW - i], lastnodes);

      // pass through the derivative of the activation function
      deriv_z = d_activation(z);

      // gets precedent weights matrix
      let lastweights = math.transpose(this.weights[this.lastW - i + 1]);

      err = math.multiply(lastweights, delta);
      delta = hadamard(err, deriv_z);

      // resizes the delta matrix to match lastnodes size
      delta = math.resize(delta, [delta.length, 1]);

      // gradient of the error with respect to the weights
      gradient = math.multiply(delta, transpose(lastnodes));

      //adjusts the weights matrix
      variation = math.multiply(this.learningRate, gradient);
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
}
