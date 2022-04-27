
// object that holds the NN settings
let options = {
  layers: [2, 5, 1],
  learningRate: 0.5,
  epochs: 200,
  debug: false,
  activation: "sigmoid"
};

let nn;
let data = [
  {
    inputs: [1, 1],
    target: [0]
  },
  {
    inputs: [0, 0],
    target: [0]
  },
  {
    inputs: [0, 1],
    target: [1]
  },
  {
    inputs: [1, 0],
    target: [1]
  }
];

// let data = [
//   {
//     inputs: [1, 1],
//     target: ["false"]
//   },
//   {
//     inputs: [0, 0],
//     target: ["false"]
//   },
//   {
//     inputs: [0, 1],
//     target: ["true"]
//   },
//   {
//     inputs: [1, 0],
//     target: ["true"]
//   }
// ];

// let data = [
  // {
  //   first: 1,
  //   second: 1,
  //   target: [0]
  // },
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
  for (let i = 0; i < 100; i++) {
    let info = data[Math.floor(Math.random() * data.length)];
    dataset.push(info);
  }

  const trainset = dataset.slice(0, 50);
  const testset = dataset.slice(50, 100);

  let accuracy = nn.test(testset);
  console.log("R² = " + accuracy);
  // console.log("accuracy before = " + accuracy + "%");

  console.log("training...");
  nn.train(trainset);

  accuracy = nn.test(testset);
  console.log("R² = " + accuracy);
}
