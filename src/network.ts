/*--------------------------------------------------------------------------

neuron-ts - A multilayer perceptron network implemented in typescript.

The MIT License (MIT)

Copyright (c) 2016 Haydn Paterson (sinclair) <haydn.developer@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

---------------------------------------------------------------------------*/

import {Model} from "./model"

export interface NetworkOptions {
  /** the learning rate (etc), typical range [0.0 .. 1.0]. defaults to 0.15. */
  rate     : number 
  /** the momentum multiplier (alpha). typical range [0.0 .. n]. defaults to 0.5. */
  momentum : number
}

/**
 * Network:
 * A multi layer perceptron network. 
 */
export class Network {

  //---------------------------------------------------------
  // todo: generalize for f(x) where c is the learning rate.
  // x = f(x)
  // d = (f(x+c) - f(x)) / x  
  //---------------------------------------------------------
  public activations = {
    "tanh": (x: number) => (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)),
  }
  public derivitives = {
    "tanh"   : (x: number) => (1 - (x * x)), // approximation needs fixing.
  }
  /** 
   * creates a new network.
   * @param {Model} the network model.
   * @returns {Network}
  */
  constructor(public model: Model, public options?: NetworkOptions) {
    this.options = this.options || {
      rate     : 0.15,
      momentum : 0.5
    }
    this.model.layers().each(layer => {
      layer.neurons().skip(1).each(neuron => {
        neuron.outputs().each(synapse => {
          synapse.weight((Math.random() - 0.5) * 2.0) 
        })
      })
    })
  }

  /**
   * forward propagates the given input through the network.
   * @param {Array<number>} the input to this network.
   * @returns {Array<number>} the networks output layer.
   */
  public forward(input: Array<number>): Array<number> {
    this.model.layers()
              .first()
              .neurons()
              .skip(1)
              .each((neuron, index) => 
                neuron.value(input[index]))
    
    this.model.layers().skip(1).each(hidden => {
      let activate = this.activations[hidden.activation()]
      hidden.neurons().each(neuron => {
        let sum = neuron.inputs().aggregate((acc, synapse) => 
            (acc + (synapse.weight() * synapse.target().value()))
        , 0)
        neuron.value( activate( sum ) )
      })
    })

    return this.model.layers()
      .last()
      .neurons()
      .skip(1)
      .select(neuron => neuron.value())
      .collect()
  }

  /**
   * trains this network via back propagation / gradient descent.
   * @param {Array<number>} the actual obtained from a call to forward()
   * @param {Array<number>} the expected value.
   * @returns {void}
   */
  public backward(actual: Array<number>, expected: Array<number>): void {
    // optional - calculate error()
    // let error = Math.sqrt(actual.reduce((acc, value, index) => {
    //   let delta = (expected[index] - value)
    //   return (acc + (delta * delta))
    // }, 0) / actual.length)
    
    this.model.layers().last().neurons().skip(1).each((neuron, index) => {
      let derive = this.derivitives[neuron.layer().activation()]
      let delta  = expected[index] - neuron.value()
      neuron.gradient(delta * derive(neuron.value()))
    })

    this.model.layers().skip(1).reverse().skip(1).each(layer => {
      let derive = this.derivitives[layer.activation()]
      layer.neurons().skip(1).each(neuron => {
        let sum = neuron.outputs().aggregate((acc, synapse) => 
          acc + (synapse.weight() * synapse.target().gradient())
        , 0) 
        neuron.gradient(sum * derive(neuron.value()))
      })
    })

    this.model.layers().skip(1).reverse().each(layer => {
      layer.neurons().skip(1).each(neuron => {
        neuron.inputs().each(synapse => {
          let olddelta = synapse.delta()
          let newdelta = 
              (this.options.rate * synapse.target().value() * neuron.gradient()) 
            + (this.options.momentum * olddelta)
          synapse.weight(synapse.weight() + newdelta)
          synapse.delta(newdelta)
        })
      })
    })
  }
}