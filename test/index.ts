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
  new neuron.Tensor(2),
  new neuron.Tensor(5, "tanh"),
  new neuron.Tensor(3, "tanh"),
  new neuron.Tensor(1, "tanh"),
]))

//------------------------------------------------------
// train the network and report every 10,000 iterations.
//------------------------------------------------------
let iteration = 0
while(iteration < 100000000) { // 100,000,000 iterations.

  // train network against xor truth table. store mean error.
  const error = (network.backward([0, 0], [0]) +
                 network.backward([0, 1], [1]) +
                 network.backward([1, 0], [1]) +
                 network.backward([1, 1], [0])) / 4

  // view approximation.
  if(iteration % 10000 === 0) {
    console.log("i:", iteration, "e:", error)
    console.log(0, network.forward([0, 0]))
    console.log(1, network.forward([0, 1]))
    console.log(1, network.forward([1, 0]))
    console.log(0, network.forward([1, 1]))
  }
  iteration++
}
