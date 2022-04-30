
// object that holds the NN settings
let options = {
  layers: [3, 5, 10, 11],
  output_labels: ["black", "white", "gray", "red", "blue", "orange", "purple", "green", "yellow", "brown", "pink"],
  learningRate: 0.5,
  epochs: 10,
  debug: false,
  activation: "sigmoid",
  max_input: 255,
  min_input: 0,
  normalize_data: true
};

// white, black, gray, red, blue, orange, purple, green, yellow, brown, pink
let nn;
let dataset = [];
let jsonObj;
// fetch("dataset.json")
//   .then(response => response.json())
//   .then(json => {dataset = json});

function preload() {
  jsonObj = loadJSON("dataset.json");
}

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

function setup() {
  noCanvas();

  for (let elem in jsonObj){
    dataset.push(jsonObj[elem]);
  }

  nn = new NeuralNetwork(options);
  console.log(dataset);
  console.log(nn.normalize(dataset));
  // dataset = [];
  // for (let i = 0; i < 100; i++) {
  //   let info = data[Math.floor(Math.random() * data.length)];
  //   dataset.push(info);
  // }

  // divides the dataset in a training part and a testing part
  let trainSlice = Math.floor(dataset.length * 0.7);

  const trainset = dataset.slice(0, trainSlice);
  const testset = dataset.slice(trainSlice, dataset.length);

  let accuracy = nn.test(testset);
  accuracy = Math.round((accuracy + Number.EPSILON) * 100) / 100;
  console.log(`accuracy:${accuracy}%`);

  console.log("training...");
  nn.train(trainset);

  accuracy = nn.test(testset);
  accuracy = Math.round((accuracy + Number.EPSILON) * 100) / 100;
  console.log(`accuracy:${accuracy}%`);

}
