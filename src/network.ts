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

import { Matrix } from "./matrix"
import { Tensor } from "./tensor"

//----------------------------------------------------------------------------------------------//
// network memory layout                                                                        //
//----------------------------------------------------------------------------------------------//
//                                                                                              //
// matrices are encoded in linear form and can be structurally thought of as..                  //
//                                                                                              //
//   i = input tensor    (3 neuron + bias)                                                      //
//   o = output tensor   (4 neuron + bias)                                                      //
//   w = weight matrix                                                                          //
//                                                                                              //
//                     (bias)                                                                   //
//                       |                                                                      //
//   i[0 ] i[1 ] i[2 ] i[3 ]                                                                    //
//     |     |     |     |                                                                      //
//   w[0 ] w[1 ] w[2 ] w[3 ] --> o[0 ]                                                          //
//   w[4 ] w[5 ] w[6 ] w[7 ] --> o[1 ]                                                          //
//   w[8 ] w[9 ] w[10] w[11] --> o[2 ]                                                          //
//   w[12] w[13] w[14] w[15] --> o[3 ]                                                          //
//                               o[4 ] -- (bias)                                                //
//                                                                                              //
// thus the weight matrix connecting input and output is [inputs * (outputs - 1)] with bias     //
// neuron encoded as the last element in the tensor.                                            //
//                                                                                              //
// if expressed as linear computation, the following would constitute a feed forward from       //
// input to output.                                                                              //
//                                                                                              //
// o[0] = o.activate( (i[0 ] * w[0 ]) + (i[1 ] * w[1 ]) + (i[2 ] * w[2 ]) + (i[3 ] * w[3 ]) )   //
// o[1] = o.activate( (i[0 ] * w[4 ]) + (i[1 ] * w[5 ]) + (i[2 ] * w[6 ]) + (i[3 ] * w[7 ]) )   //
// o[2] = o.activate( (i[0 ] * w[8 ]) + (i[1 ] * w[9 ]) + (i[2 ] * w[10]) + (i[3 ] * w[11]) )   //
// o[3] = o.activate( (i[0 ] * w[12]) + (i[1 ] * w[13]) + (i[2 ] * w[14]) + (i[3 ] * w[15]) )   //
//                                                                                              //
//----------------------------------------------------------------------------------------------//

export interface Kernel {
  input: Tensor,
  output: Tensor,
  matrix: Matrix
}
export class Network {
  public matrices: Array<Matrix>
  public kernels : Array<Kernel>
  private output : Array<number>
  /**
   * creates a new network with the given tensor layers.
   * @param {Tensor[]} tensors the tensors for each layer in the network.
   * @returns {Network}
   */
  constructor(public tensors: Tensor[]) {
    // initialize output buffer.
    this.output = new Array(this.tensors[this.tensors.length - 1].data.length - 1)
    // initialize network matrices.
    this.matrices = new Array<Matrix>(this.tensors.length - 1)
    for (let i = 0; i < this.tensors.length - 1; i++) {
      this.matrices[i] = new Matrix (
        this.tensors[i + 0].data.length,
        this.tensors[i + 1].data.length - 1
      )
    }
    // initialize network compute kernels.
    this.kernels = new Array(this.matrices.length)
    for (let i = 0; i < this.kernels.length; i++) {
      this.kernels[i] = {
        input : this.tensors[i + 0],
        output: this.tensors[i + 1],
        matrix: this.matrices[i]
      }
    }
  }
  /**
   * returns the memory footprint of this network in bytes.
   * @returns {number}
   */
  public memory(): number {
    const tensors  = this.tensors.reduce((acc, t) => acc + (t.data.byteLength), 0)
    const matrices = this.matrices.reduce((acc, m) => acc + (m.data.byteLength), 0)
    return tensors + matrices
  }
  /**
   * returns the number of inputs accepted by this network.
   * @returns {number}
   */
  public inputs(): number {
    return (this.tensors[0].data.length - 1)
  }
  /**
   * returns the number of outputs from this network.
   * @returns {number}
   */
  public outputs(): number {
    return (this.tensors[this.tensors.length - 1].data.length - 1)
  }
  /**
   * executes this network, propagating input to output.
   * @param {Array<number>} input the input buffer to write to the network.
   * @returns {Array<number>} the outputs for this network.
   */
  public forward(input: Array<number>): Array<number> {
    // load data from input.
    for (let i = 0; i < input.length; i++) {
      this.kernels[0].input.data[i] = input[i]
    }
    // feed forward values through the network.
    for (let k = 0; k < this.kernels.length; k++) {
      const kernel = this.kernels[k]
      for (let o = 0; o < kernel.matrix.outputs; o++) {
        let sum = 0
        for (let i = 0; i < kernel.matrix.inputs; i++) {
          sum += kernel.matrix.get(i, o) * kernel.input.data[i]
        }
        kernel.output.data[o] = kernel.output.activation.activate(sum)
      }
    }
    // unload output layer return value.
    for (let o = 0; o < this.output.length; o++) {
      this.output[o] = this.kernels[this.kernels.length - 1].output.data[o]
    } return this.output
  }
}