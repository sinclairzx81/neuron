### neuron

neural network implemented in javascript.

### overview

neuron is a javascript implementation of a multi layer perceptron network. The library allows one to quickly create fully connected feed forward networks and iteratively train them to approximate some desired output.

neuron was written to allow for fast, interactive training in the browser for small networks. The library is offered as is for anyone who finds it useful, educational or just interesting.

### building the project
```
npm install typescript -g
npm install typescript-bundle -g
npm run build-test
node bin/test
```

### approximate xor

the following code sets up a network and iteratively trains it to approximate xor. This network is using tanh activation per each layer, with each layer given an additional bias neuron with a value of 1.0.

```typescript
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
    console.log("-", iteration, error)
    console.log(0, network.forward([0, 0]))
    console.log(1, network.forward([0, 1]))
    console.log(1, network.forward([1, 0]))
    console.log(0, network.forward([1, 1]))
  }
  iteration++
}
```