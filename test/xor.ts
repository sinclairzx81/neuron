import * as neuron from "../src/index"

// 3 layer network.
let network = new neuron.Network(new neuron.Model([
  {neurons: 2},
  {neurons: 2, activation: "tanh"},
  {neurons: 1, activation: "tanh"}
]))

// XOR training set.
let examples = [
  {input: [0, 0], expect: [0]},
  {input: [1, 0], expect: [1]},
  {input: [0, 1], expect: [1]},
  {input: [1, 1], expect: [0]}
]

let epochs = 1024

// train network
for(let i = 0; i < epochs; i++) {
  examples.forEach(example => {
    let actual = network.forward(example.input)
    network.backward(actual, example.expect)
  })
}

// show result.
console.log(`results after [${epochs * examples.length}] epochs`)
examples.forEach(example => {
  let actual = network.forward(example.input)
  console.log("input: ", example.input, 
              "ideal:",  example.expect, 
              "actual:", actual)
})
