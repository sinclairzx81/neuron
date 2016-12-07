
# neuron-ts

A multilayer perceptron network implemented in typescript.

## overview

This project is a quick implementation of a multilayer perceptron network (MLP) that serves
as a test bed for exploring various machine learning implementations and abstractions in 
javascript.

## building the project

```
npm install typescript -g
tsc -p tsconfig.json
```

## creating a network

The following creates a simple MLP to solve the XOR problem. This problem is 
interesting in the sense that XOR gives an output which is non linearly separable. The goal
of the program is to teach the network to approximate XOR. (or any other binary operators)

Additional information on the XOR problem can be found [here](http://www.mind.ilstu.edu/curriculum/artificial_neural_net/xor_problem_and_solution.php). 

The following sets up the minimum network requirements:
- 2 neuron input layer 
- 2 neuron hidden layer
- 1 neuron output layer 
- The hidden and output layers are activated with the hyperbolic tangent transfer function (tanh).

```typescript 
import * as neuron from "./neuron/index"

let network = new neuron.Network(new neuron.Model([
  {neurons: 2},                        
  {neurons: 2, activation: "tanh"}, 
  {neurons: 1, activation: "tanh"}
]))
```

## training the network

Once a network is in place, you can pass it values immediately with the ```forward(input)``` method. This method will propagate the 
value you pass through the network (using random weights) and return a (nonsense) result. 

```typescript
let result = network.forward([0, 1])
```
To train the network, you need to pass the ```actual``` value back to the network on the ```backward(actual, expected)``` method. This results 
in the error between the ```actual``` and ```expected``` being back propagated through the network, and the network weights are adjusted 
to reduce the error using gradient descent. This process needs to be repeated for a number of iterations.

```typescript
for(let i = 0; i < 1024; i++) {
  network.backward(network.forward[0, 0], [0])
  network.backward(network.forward[0, 1], [1])
  network.backward(network.forward[1, 0], [1])
  network.backward(network.forward[1, 1], [0])
}
```
once finished, the ```actual``` value given from ```forward()``` should now converge to something 
similar to the following output.
```
input:  [ 0, 0 ] ideal: [ 0 ] actual: [ 0.011208046227693558 ]
input:  [ 1, 0 ] ideal: [ 1 ] actual: [ 0.9938235878944397 ]
input:  [ 0, 1 ] ideal: [ 1 ] actual: [ 0.9938225746154785 ]
input:  [ 1, 1 ] ideal: [ 0 ] actual: [ 0.008087895810604095 ]
```
