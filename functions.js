/* ACTIVATION FUNCTIONS */

// sigmoid function and its derivative
// f(x) = 1/1+e^-x
function sigmoid(x) {
  return 1 / (1 + math.exp(-x));
}

// derivative of sigmoid
// f'(x) = f(x)*(1-f(x))
function d_sigmoid(x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

// Rectified Linear Units
function ReLU(x) {
  if (x < 0) return 0;
  else return x;
}

function d_ReLU(x) {
  if (x < 0) return 0;
  else return 1;
}
// softplus = ln(1+e^-x)
function softplus(x) {
  return Math.log(1 + math.exp(-x));
}
// the derivative of the sofplus function is the sigmoid function
function d_softplus(x) {
  return sigmoid(x);
}

// gaussian function: e^(-x²)
function gaussian(x) {
  return math.exp(-1 * math.pow(x, 2));
}

function d_gaussian(x) {
  return -2 * x * math.exp(-1 * math.pow(x, 2));
}

// softmax activation/probability function
function softmax(matrix) {
  let total = 0;
  for (let i = 0; i < math.size(matrix)[0]; i++) {
    total += math.exp(math.subset(matrix, math.index(i)));
  }

  return math.map(matrix, (value, index, matrix) => {
    return math.exp(value) / total;
  });
}

// derivative of softmax: f'(x)= f(x)*(1-f(x))
function d_softmax(matrix) {
  return math.multiply(softmax(matrix), math.subtract(1, softmax(matrix)));
}


/* UTILITIES */

// return a matrix with every element passed through the function
function activation(matrix, f_name) {
  f_name = f_name.toLowerCase();
  let activated = matrix.map((value, index, matrix) => {
    if (f_name === "relu") {
      return ReLU(value);
    } else if (f_name === "softplus") {
      return softplus(value);
    } else if (f_name === "gaussian") {
      return gaussian(value);
    } else if (f_name === "sine") {
      return Math.sin(value);
    } else if (f_name === "sigmoid"){
      return sigmoid(value);
    } else {
      console.error("unknown activation function")
    }
  });

  if (f_name === "softmax") {
    activated = d_softmax(matrix);
  }
  return activated;
}

function d_activation(matrix, f_name) {
  f_name = f_name.toLowerCase();
  let activated = matrix.map((value, index, matrix) => {
    if (f_name === "relu") {
      return d_ReLU(value);
    } else if (f_name === "softplus") {
      return d_softplus(value);
    } else if (f_name === "gaussian") {
      return d_gaussian(value);
    } else if (f_name === "sine") {
      return Math.cos(value);
    } else if (f_name === "sigmoid"){
      return d_sigmoid(value);
    }
  });

  if (f_name === "softmax") {
    activated = d_softmax(matrix);
  }
  return activated;
}

function hadamard(a, b) {
  let res = a.map((value, index, matrix) => {
    return value * math.subset(b, math.index(index));
  });
  return res;
}

// transpose function
function transpose(matrix) {
  // if matrix is an array
  if (matrix instanceof Array) {
    let res = [];
    let rows = matrix[0].length || 1;
    let columns = matrix.length;

    for (let i = 0; i < rows; i++) {
      let tmp = [];
      for (let j = 0; j < columns; j++) {
        if (rows == 1) {
          tmp.push(matrix[j]);
        } else {
          tmp.push(matrix[j][i]);
        }
      }
      res.push(tmp);
    }
    return res;
  } else if (math.size(matrix).length == 1) {
    // if the matrix is a X by 1 vector, math.transpose is useless
    // so i used this math.resize function
    return math.resize(matrix, [1, math.size(matrix)[0]]);
  } else {
    // didn't find any other edge cases, so if ultimately is a non-vector
    // math.matrix object, just uses the ordinary transpose function
    return math.transpose(matrix);
  }
}

// simple (and maybe inefficient) shuffle function
function data_shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    let j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function getAverage(dataset) {
  // creates a vector that contains the summation of all data targets
  let sum = math.zeros(dataset[0].target.length);
  for (let data of dataset) {
    sum = math.add(sum, data.target);
  }

  return math.multiply(sum, 1 / dataset.length);
}

// total sum of squares
function TSS(targets, means) {
  // error = (target-mean of targets)²
  if (targets instanceof math.matrix) {
    let errors = math.map(targets, (value, index, matrix) => {
      let mean = math.subset(means, math.index(index));
      return math.pow(value - mean, 2);
    });
    return errors;
  } else if (targets instanceof Array) {
    targets = math.matrix(targets);

    let errors = math.map(targets, (value, index, matrix) => {
      let mean = math.subset(means, math.index(index));
      return math.pow(value - mean, 2);
    });

    return errors;
  } else {
    return math.pow(targets - means, 2);
  }
}

// error sum of squares
function ESS(targets, results) {
  // error = (target-guess)²
  if (results instanceof math.matrix) {
    let errors = math.map(results, (value, index, matrix) => {
      let target = math.subset(targets, math.index(index));
      return math.pow(target - value, 2);
    });
    return errors;
  } else if (results instanceof Array) {
    results = math.matrix(results);
    targets = math.matrix(targets);

    let errors = math.map(results, (value, index, matrix) => {
      let target = math.subset(targets, math.index(index));
      return math.pow(target - value, 2);
    });

    return errors;
  } else {
    return math.pow(targets - results, 2);
  }
}
