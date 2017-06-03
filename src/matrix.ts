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

/**
 * A matrix type used to express the weights between layers of a network.
 */
export class Matrix {
  public data: Float64Array
  /**
   * creates a new matrix with the given input and output dimensions (width and height)
   * @param {number} inputs the number of inputs.
   * @param {number} outputs the number of outputs.
   * @returns {Matrix}
   */
  constructor(public inputs: number, public outputs: number) {
    this.data = new Float64Array(this.inputs * this.outputs)
  }
  /**
   * gets a value within this matrix.
   * @param {number} i the input value
   * @param {number} o the output value.
   * @returns {number}
   */
  public get(i: number, o: number): number {
    return this.data[i + (o * this.inputs)]
  }
  /**
   * sets a value within this matrix.
   * @param {number} i the input index.
   * @param {number} o the output index.
   * @param {number} value the value to set.
   */
  public set(i: number, o: number, value: number): void {
    this.data[i + (o * this.inputs)] = value
  }
}