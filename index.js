
// object that holds the NN settings
let options = {
  layers: [2, 4, 2],
  learningRate: 0.5,
  // input_labels: ["first", "second"],
  output_labels: ["true", "false"],
  // activation: ["sigmoid", "sigmoid"],
  epochs: 200
};

let nn;
// let data = [
//   {
//     inputs: [1, 1],
//     target: [0]
//   },
//   {
//     inputs: [0, 0],
//     target: [0]
//   },
//   {
//     inputs: [0, 1],
//     target: [1]
//   },
//   {
//     inputs: [1, 0],
//     target: [1]
//   }
// ];

let data = [
  {
    inputs: [1, 1],
    target: ["false"]
  },
  {
    inputs: [0, 0],
    target: ["false"]
  },
  {
    inputs: [0, 1],
    target: ["true"]
  },
  {
    inputs: [1, 0],
    target: ["true"]
  }
];

// let data = [
//   {
//     first: 1,
//     second: 1,
//     target: [0]
//   },
//   {
//     first: 0,
//     second: 0,
//     target: [0]
//   },
//   {
//     first: 0,
//     second: 1,
//     target: [1]
//   },
//   {
//     first: 1,
//     second: 0,
//     target: [1]
//   }
// ];

let dataset;
function setup() {
  noCanvas();

  nn = new NeuralNetwork(options);

  dataset = [];
  for (let i = 0; i < 50; i++) {
    let info = data[Math.floor(Math.random() * data.length)];
    dataset.push(info);
  }

  let accuracy = nn.test(dataset);
  console.log("accuracy random = " + accuracy + "%");
  // console.log("accuracy before = " + accuracy + "%");

  nn.train(dataset);

  accuracy = nn.test(dataset);
  console.log("accuracy trained = " + accuracy + "%");

  let state = nn.saveState();
  nn.reset();

  accuracy = nn.test(dataset);
  console.log("accuracy reset = " + accuracy + "%");

  nn.loadState(state);

  accuracy = nn.test(dataset);
  console.log("accuracy loaded state = " + accuracy + "%");

  // console.log("accuracy after = " + accuracy + "%");
}
