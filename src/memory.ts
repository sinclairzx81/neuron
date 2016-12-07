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

export interface Layer {
  index      : number,
  activation : string,
  neurons    : Array<number>
}
export interface Synapse {
  index      : number,
  output     : number,
  input      : number,
  weight     : number
}
export interface Neuron {
  index      : number,
  layer      : number,
  inputs     : Array<number>,
  outputs    : Array<number>
}
export interface MemoryLayerDescriptor {
  neurons     : number,
  activation? : string
}

/**
 * Memory:
 * 
 * Low level memory layout for a multi layer perceptron network.
 * This class is responsible for provisioning the network, setting
 * up memory buffers and providing indices into those buffer.
 */
export class Memory {
  //-------------------------------
  // indices.
  //-------------------------------
  public layers     : Array<Layer>
  public synapses   : Array<Synapse>
  public neurons    : Array<Neuron>
  //-------------------------------
  // neuron values.
  //-------------------------------
  public values     : Float32Array
  public gradients  : Float32Array
  //------------------------------
  // synapse values.
  //------------------------------
  public weights    : Float32Array
  public deltas     : Float32Array

  /**
   * creates a new model.
   * @param {Array<ModelDescriptor>} 
   */
  constructor(descriptors: Array<MemoryLayerDescriptor>) {
    let neuron_index  = 0
    let synapse_index = 0
    //---------------------------------------------
    // create layer indices
    //---------------------------------------------
    neuron_index = 0
    this.layers = descriptors.map((descriptor, index) => {
      let layer = {
        index     : index,
        activation: descriptor.activation,
        neurons   : []
      }
      for(let i = 0; i < (descriptor.neurons + 1); i++) {
        layer.neurons.push(neuron_index)
        neuron_index += 1;
      } return layer
    })
    //---------------------------------------------
    // create neuron indices.
    //---------------------------------------------
    this.neurons = []
    neuron_index = 0;
    for(let i = 0; i < this.layers.length; i++) {
      for(let j = 0; j < this.layers[i].neurons.length; j++) {
        this.neurons.push({
          index    : neuron_index,
          layer    : i,
          inputs   : [],
          outputs  : []
        }); neuron_index += 1;
      }
    }
    //---------------------------------------------
    // create synapse indices.
    //---------------------------------------------
    this.synapses = []
    synapse_index = 0
    for(let i = 0; i < (this.layers.length - 1); i++) {
      let input  = this.layers[i + 0]
      let output = this.layers[i + 1]
      for(let output_idx = 1; output_idx < output.neurons.length; output_idx += 1 ) {
        for(let input_idx = 0; input_idx < input.neurons.length; input_idx += 1 ) {
          this.synapses.push({
            index  : synapse_index,
            output : input.neurons [input_idx],
            input  : output.neurons[output_idx],
            weight : synapse_index
          }); synapse_index += 1;
        }
      }
    }

    //---------------------------------------------
    // the neuron input and output synapse values.
    //---------------------------------------------
    for(let i = 0; i < this.synapses.length; i += 1) {
      let input   = this.neurons[this.synapses[i].output]
      let output  = this.neurons[this.synapses[i].input]
      input.outputs.push(this.synapses[i].index)
      output.inputs.push(this.synapses[i].index)
    }
    //---------------------------------------------
    // create neuron buffers
    //---------------------------------------------
    this.values    = new Float32Array(this.neurons.length)
    this.gradients = new Float32Array(this.neurons.length)

    //---------------------------------------------
    // create synapse buffers.
    //---------------------------------------------
    this.weights  = new Float32Array(this.synapses.length)
    this.deltas   = new Float32Array(this.synapses.length)

    //---------------------------------------------
    // setup bias values.
    //---------------------------------------------
    for(let i = 0; i < this.layers.length; i += 1) {
      let layer = this.layers[i]
      let neuron = this.neurons[layer.neurons[0]]
      this.values[neuron.index] = 1.0
      for(let j = 0; j < neuron.outputs.length; j += 1) {
        let synapse = this.synapses[neuron.outputs[j]]
        this.weights[synapse.index] = 1.0
      }
    }  
  }
}