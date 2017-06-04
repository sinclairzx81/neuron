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

type Activation = {
  activate: (x: number) => number
  derive  : (x: number) => number
}

/**
 * selects an activation function.
 * @param {string} type the type of activation.
 * @returns {Activation}
 */
const select = (type: string): Activation => {
  switch(type) {
    case "identity": return  {
      activate: x => x,
      derive  : x => 1
    }
    case "tanh": return  {
      activate: x => (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)),
      derive  : x => (1 - (x * x))
    }
    case "binary-step": return {
      activate: x => (x >= 0) ? 1 : 0,
      derive  : x => (x >= 0) ? 1 : 0
    }
    case "relu": return {
      activate: x => (x >= 0) ? x : 0,
      derive  : x => (x >= 0) ? 1 : 0
    }
    default: throw Error("unknown activation")
  }
}

/** An n-dimensional vector used to represent a layer of a network.  */
export class Tensor {
  public data      : Float64Array
  public activation: Activation
  /**
   * creates a new tensor.
   * @param {number} units the number of units in this tensor (will result in +1 to include the bias)
   * @param {Activation} activation the activation functions to use. (defaults to "identity")
   * @param {number} bias the value of the tensors bias neuron (defaults to 1.0)
   * @returns {Tensor}
   */
  constructor(units: number, activation: string = "identity", bias: number = 1.0) {
    this.data = new Float64Array(units + 1)
    this.data[this.data.length - 1] = bias
    this.activation = select(activation)
  }
}