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

import {Sequence} from "./sequence"
import {Memory}   from "./memory"

/**
 * Synapse: 
 * 
 * A synpase is obtained from a neurons input / output. 
 * The synapse is always constructed relative to the 
 * neuron that requested it. The target() on this 
 * type therefore may refer to the input neuron, or
 * the output neuron depending on how the synapse 
 * was requested.
 */
export class Synapse {

  /**
   * creates a new synapse.
   * @param {Memory} the memory map.
   * @param {number} the synapse index.
   * @param {number} the target neuron index.
   * @returns {Synapse}
   */
  constructor(
    private memory: Memory, 
    private index: number, 
    private targetIndex: number)  { }
  
  /**
   * gets or sets the weight for this synapse.
   * @param {number?} optional value to set.
   * @returns {number}
   */
  public weight(value?: number): number {
    let synapse = this.memory.synapses[this.index]
    if(value !== undefined) this.memory.weights[synapse.weight] = value
    return this.memory.weights[synapse.weight]
  }

  /**
   * gets or sets the delta weight for this synapse.
   * @param {number?} optional value to set.
   * @returns {number}
   */
  public delta(value?: number): number {
    let synapse = this.memory.synapses[this.index]
    if(value !== undefined) this.memory.deltas[synapse.weight] = value
    return this.memory.deltas[synapse.weight]
  }

  /**
   * returns the target neuron for this synapse. If this
   * synapse was returned from a neurons inputs, then this
   * target refers to the previous layer. If the synapse
   * was requested for a neurons output, the this target
   * points to a neuron in the next layer.
   * @returns {Neuron}
   */
  public target(): Neuron {
    return new Neuron(this.memory, this.targetIndex)
  }
}

/**
 * Neuron:
 * A simple neuron, provides navigational function
 * from this neuron to its respective inputs and 
 * outputs (by way of a synapse), and to the layer
 * and allows the caller to get and set the values 
 * contained within.
 */
export class Neuron {

  /**
   * creates a new neuron.
   * @param {Memory} the memory data structure.
   * @param {number} the memory index for this neuron.
   * @returns {Neuron}
   */
  constructor(
    private memory: Memory, 
    private index: number) {
  }

  /**
   * returns true if this neuron is a bias neuron.
   * @return {boolean}
   */
  public bias(): boolean {
    let neuron = this.memory.neurons[this.index]
    let layer  = this.memory.layers[neuron.layer] 
    return layer.neurons[0] === neuron.index
  }

  /**
   * returns the layer that this neuron resides.
   * @returns {Layer}
   */
  public layer(): Layer {
    let index = this.memory.neurons[this.index]
    return new Layer(this.memory, index.layer)
  }

  /**
   * gets or sets the value stored in this neuron.
   * @param {number?} optional value to set this neuron to.
   * @return {number}
   */
  public value(value?: number): number {
    if(value !== undefined) this.memory.values[this.index] = value
    return this.memory.values[this.index]
  }

  /**
   * gets or sets this neurons gradient value.
   * @param {number?} optional value to set this neurons gradient to.
   * @returns {number}
   */
  public gradient(value?: number): number {
    if(value !== undefined) this.memory.gradients[this.index] = value
    return this.memory.gradients[this.index]
  }

  /**
   * returns the input synapses for this neuron. 
   * @param {Sequence<Synapse>}
   */
  public inputs(): Sequence<Synapse> {
    return new Sequence<Synapse>(this.memory.neurons[this.index].inputs.map(index => {
      let synapse = this.memory.synapses[index]
      return new Synapse(this.memory, synapse.index, synapse.output)
    }))
  }

  /**
   * returns the output synapses for this neuron.
   * @returns {Sequence<Synapse>}
   */
  public outputs(): Sequence<Synapse> {
    return new Sequence<Synapse>(this.memory.neurons[this.index].outputs.map(index => {
      let synapse = this.memory.synapses[index]
      return new Synapse(this.memory, synapse.index, synapse.input)
    }))
  }
}

/**
 * Layer
 * Represents a layer in the network and provides 
 * nanivational functions to the layers neurons and
 * previous and next layers.
 */
export class Layer {
  /**
   * creates a new layer.
   * @param {Memory} the memory map.
   * @param {number} the layer index.
   * @returns {Layer}
   */
  constructor(
    private memory: Memory, 
    private index: number) 
    { }
  
  /**
   * returns the type of this layer.
   * @returns {string}
   */
  public type(): string {
    if(this.index === 0) {
      return "input"
    } else if(this.index === (this.memory.layers.length - 1)) {
      return "output"
    } else {
      return "hidden"
    }
  }

  /**
   * returns the next layer or undefined if output layer.
   * @returns {Layer}
   */
  public next(): Layer {
    if((this.index + 1) < this.memory.layers.length) {
      return new Layer(this.memory, this.index + 1)
    } return undefined
  } 

  /**
   * returns the previous layer or undefined if input layer.
   * @returns {Layer}
   */
  public previous(): Layer {
    if(this.index > 0) {
      return new Layer(this.memory, this.index - 1)
    } return undefined
  }

  /**
   * returns the activation type for this layer.
   * @returns {string}
   */
  public activation(): string {
    return this.memory.layers[this.index].activation
  }

  /**
   * returns the neurons in this layer.
   * @returns {Sequence<Neuron>}
   */
  public neurons(): Sequence<Neuron> {
    return new Sequence<Neuron>(
      this.memory.layers[this.index].neurons.map((neuron, index) => {
        return new Neuron(this.memory, this.memory.neurons[neuron].index)
      })
    )
  }
}

export interface ModelLayerDescriptor {
  neurons     : number,
  activation? : string
}
/**
 * Model
 * A container type for a multi layered perceptron network.
 */
export class Model {
  public memory: Memory
  constructor(private descriptors: Array<ModelLayerDescriptor>) {
    this.memory = new Memory(descriptors)
  }

  /**
   * returns the layers in this network.
   * @returns {Sequence<Layer>}
   */
  public layers(): Sequence<Layer> {
    return new Sequence<Layer>(this.memory.layers.map((layer) => {
      return new Layer(this.memory, layer.index)
    }))
  }
}