/*--------------------------------------------------------------------------

neuron - neural network written in javascript.

The MIT License (MIT)

Copyright (c) 2017 Haydn Paterson (sinclair) <haydn.developer@gmail.com>

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

import { Tensor }  from "./tensor"
import { Matrix }  from "./matrix"
import { Network } from "./network"

/**
 * creates a sampling range function for initializing weights. The min and max
 * should be the same value (with min negated), to achieve a zero sum distribution. 
 * This seems to yield slightly better results during gradient descent 
 * @param {number} resolution the resolution (should be set to the input length)
 * @param {number} min the min value (-0.5 is good.)
 * @param {number} max the max value ( 0.5 is good.)
 * @returns {Function}
 */
const sampler = (resolution: number, min: number, max: number) => {
  const w = max - min
  const d = w / (resolution - 1)
  switch(resolution) {
    case 0:  throw new Error("0 sampler resolution not allowed.")
    case 1:  return (offset) => min + ((max - min) / 2)
    default: return (offset) => min + (d * offset)
  }
}

export interface Kernel {
  input : { tensor: Tensor, grads:  Float64Array },
  output: { tensor: Tensor, grads:  Float64Array },
  matrix: { matrix: Matrix, deltas: Matrix}
}
/**
 * A network trainer that uses classic back propagation / gradient descent to 
 * train a network. The class also proxies to the underlying network for 
 * convenience.
 */
export class Trainer {
  private gradients: Array<Float64Array>
  private deltas   : Array<Matrix>
  private kernels  : Array<Kernel>

  /**
   * creates a new trainer.
   * @param {Network} network the network to train.
   * @param {number} step the network step size (defaults to 0.15)
   * @param {number} momentum the network momenum (defaults to 0.5)
   * @returns {Trainer}
   */
  constructor(public network: Network, private step: number = 0.15, private momentum: number = 0.5) {
    // initialize matrix deltas
    this.deltas = new Array<Matrix>(this.network.matrices.length)
    for (let i = 0; i < this.network.matrices.length; i++) {
      this.deltas[i] = new Matrix(
        this.network.matrices[i].inputs,
        this.network.matrices[i].outputs
      )
    }
    // initialize neuron gradients.
    this.gradients = new Array(this.network.tensors.length)
    for (let i = 0; i < this.network.tensors.length; i++) {
      this.gradients[i] = new Float64Array(this.network.tensors[i].data.length)
    }
    // setup weight distribution
    for (let m = 0; m < this.network.matrices.length; m++) {
      const sample = sampler(this.network.matrices[m].inputs, -0.5, 0.5)
      for (let o = 0; o < this.network.matrices[m].outputs; o++) {
        for (let i = 0; i < this.network.matrices[m].inputs; i++) {
          this.network.matrices[m].set(i, o, sample(i))
        }
      }
    }
    // initialize compute kernels.
    this.kernels = new Array(this.network.kernels.length)
    for (let i = 0; i < this.network.kernels.length; i++) {
      this.kernels[i] = {
        matrix: {
          matrix: this.network.matrices[i],
          deltas: this.deltas[i]
        },
        input: {
          tensor: this.network.tensors[i + 0],
          grads: this.gradients[i + 0]
        },
        output: {
          tensor: this.network.tensors[i + 1],
          grads: this.gradients[i + 1]
        },
      }
    }
  }
  /**
   * (proxied) executes this network, propagating input to output.
   * @param {Array<number>} input the input buffer to write to the network.
   * @returns {Array<number>} the outputs for this network.
   */
  public forward(input: Array<number>): Array<number> {
    return this.network.forward(input)
  }
  /**
   * computes the error for this network.
   * @param {Array<number>} input the network input.
   * @param {Array<number>} expect the expected output.
   * @returns {number} the error.
   */
  public error(input: Array<number>, expect: Array<number>): number {
    const actual = this.network.forward(input)
    return Math.sqrt(actual.reduce((acc, value, index) => {
      let delta = (expect[index] - value)
      return (acc + (delta * delta))
    }, 0) / actual.length)
  }
  /**
   * trains the network.
   * @param {Array<number>} input the network input.
   * @param {Array<number>} expect the expected output.
   * @returns {void}
   */
  public backward(input: Array<number>, expect: Array<number>): void {
    // phase 0: execute the network, write to output layer.
    this.network.forward(input)

    // phase 1: calculate output layer gradients.
    const kernel = this.kernels[this.kernels.length - 1]
    for (let o = 0; o < kernel.matrix.matrix.outputs; o++) {
      const delta = (expect[o] - kernel.output.tensor.data[o])
      kernel.output.grads[o] = (delta * kernel.output.tensor.activation.derive(kernel.output.tensor.data[o]))
    }

    // phase 2: calculate gradients on hidden layers.
    for (let k = this.kernels.length - 1; k > 0; k--) {
      const kernel = this.kernels[k]
      for (let i = 0; i < kernel.matrix.matrix.inputs; i++) {
        let delta = 0
        for (let o = 0; o < kernel.matrix.matrix.outputs; o++) {
          delta += kernel.matrix.matrix.get(i, o) * kernel.output.grads[o]
        }
        kernel.input.grads[i] = (delta * kernel.input.tensor.activation.derive(kernel.input.tensor.data[i]))
      }
    }
    // phase 3: gradient decent on the weights.
    for (let k = this.kernels.length - 1; k > -1; k--) {
      const kernel = this.kernels[k]
      for (let i = 0; i < kernel.matrix.matrix.inputs; i++) {
        for (let o = 0; o < kernel.matrix.matrix.outputs; o++) {  
          const old_delta  = kernel.matrix.deltas.get(i, o)
          const new_delta  = (this.step * kernel.input.tensor.data[i] * kernel.output.grads[o]) + (this.momentum * old_delta)
          const new_weight = kernel.matrix.matrix.get(i, o) + new_delta
          kernel.matrix.matrix.set(i, o, new_weight)
          kernel.matrix.deltas.set (i, o, new_delta)
        }
      }
    }
  }
}