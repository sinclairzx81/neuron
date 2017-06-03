import * as neuron from "../src/index"

//------------------------------------------------------
//  network topology
//
//    0 0     <--- input layer
//  / /|\ \
// 0 0 0 0 0  <--- hidden layer 0
//  \ \|/ /        
//   0 0 0    <--- hidden layer 1
//    \|/
//     0      <--- output layer
//
//------------------------------------------------------

const network = new neuron.Trainer(new neuron.Network([
  new neuron.Tensor(2, 1.0),
  new neuron.Tensor(6, 1.0, neuron.activate.tanh),
  new neuron.Tensor(6, 1.0, neuron.activate.tanh),
  new neuron.Tensor(1, 1.0, neuron.activate.tanh)
]))

//------------------------------------------------------
// train the network and report every 10,000 iterations.
//------------------------------------------------------
let iteration = 0
while(iteration < 10000000) { // 10,000,000 iterations.

  // train network against xor truth table.
  network.backward([0, 0], [0])
  network.backward([0, 1], [1])
  network.backward([1, 0], [1])
  network.backward([1, 1], [0])

  // view approximation.
  if(iteration % 10000 === 0) {
    console.log("-", iteration)
    console.log(0, network.forward([0, 0]))
    console.log(1, network.forward([0, 1]))
    console.log(1, network.forward([1, 0]))
    console.log(0, network.forward([1, 1]))
  }
  iteration++
}