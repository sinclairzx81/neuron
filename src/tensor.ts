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

import {Activation} from "./activate"

/**
 * An n-dimensional vector used to represent a layer of a network.
 */
export class Tensor {
  public data: Float64Array
  /**
   * creates a new tensor.
   * @param {number} units the number of units in this tensor (will result in +1 to include the bias)
   * @param {number} bias the bias value for this tensor (defaults to 1.0)
   * @param {Activation} activation the activate and derive functions. (defaults to identity)
   * @returns {Tensor}
   */
  constructor(units: number, bias: number = 1.0, public activation: Activation = { activate: x=>x, derive: x=>1 }) {
    this.data = new Float64Array(units + 1)
    this.data[this.data.length - 1] = bias
  }
  /**
   * the bias value for this tensor.
   * @returns {number}
   */
  public bias(): number {
    return this.data[this.data.length - 1]
  }
}