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

/**
 * Sequence: query operators over sequences.
 */
export class Sequence<T> {
  /**
   * creates a new Sequence from the given array.
   * @param {Array<T>} the array to query.
   * @returns {Sequence<T>}
   */
  constructor(private array: Array<T>) { }

  /**
   * Preforms a typical reduce with accumulator across this sequence.
   * @param {Function} the aggregate function.
   * @returns {U}
   */
  public aggregate<U>(func: (acc: U, value: T, index: number, array: Array<T>) => U, initial?: U): U {
      let acc = initial
      for(let i = 0; i < this.array.length; i++) {
         acc = func(acc, this.array[i], i, this.array)
      } return acc
  }

  /**
   * Filters the sequence for the given expression.
   * @param {Function} the filter function.
   * @returns {Sequence<T>}
   */
  public where(func: (value: T, index: number, array: Array<T>) => boolean): Sequence<T> {
    let buffer = new Array<T>(this.array.length)
    let count  = 0
    for(let i = 0; i < this.array.length; i++) {
      if(func(this.array[i], i, this.array) === true) {
        buffer[count] = this.array[i]
        count += 1
      }
    } return new Sequence<T>(buffer.slice(0, count))
  }

  /**
   * Preforms a typical map across this sequence, and returns
   * the new mapped sequence.
   * @param {Function} the mapping function.
   * @returns {Sequence<U>}
   */
  public select<U>(func: (value: T, index: number, array: Array<T>) => U): Sequence<U> {
    let buffer = new Array<U>(this.array.length)
    for(let i = 0; i < this.array.length; i++) {
      buffer[i] = func(this.array[i], i, this.array)
    } return new Sequence<U>(buffer)
  }

  /**
   * Preforms a typical flat map across this sequence and returns
   * the new mapped sequence.
   * @param {Function} the map function.
   * @returns {Sequence<U>}
   */
  public selectMany<U>(func: (value: T, index: number, array: Array<T>) => Array<U>): Sequence<U> {
    let gather = new Array<Array<U>>()
    for(let i = 0; i < this.array.length; i++) {
      gather.push(func(this.array[i], i, this.array))
    }
    let buffer = new Array<U>(gather.reduce((acc, n) => acc + n.length, 0))
    let index = 0;
    for(let i = 0; i < gather.length; i++) {
      for(let j = 0; j < gather[i].length; j++) {
        buffer[index] = gather[i][j]
        index += 1
      }
    } return new Sequence<U>(buffer)
  }

  /**
   * Iterates through each element in the sequence, allowing the 
   * caller to mutate each element. The ability to mutate depends
   * on if the element is a reference type, otherwise no action
   * is taken. This function returns itself for additional chaining.
   * @param {Function} the iteration function.
   * @returns {Sequence<T>}
   */
  public each(func: (value: T, index: number, array: Array<T>) => void): Sequence<T> {
    for(let i = 0; i < this.array.length; i++) {
      func(this.array[i], i, this.array)
    } return this
  }

  /**
   * Counts the number of elements.
   * @returns {number}
   */
  public count(): number {
    return this.array.length
  }

  /**
   * Returns the first value in this sequence. If this sequence
   * has no elements this method will throw an error.
   * @returns {T}
   */
  public first(): T {
    if(this.array.length === 0) {
      throw Error("no elements exist.")
    } else {
      return this.array[0]
    }
  }

  /**
   * Returns the last value in this sequence. If this sequence
   * has no elements this method will throw an error.
   * @returns {T}
   */
  public last(): T {
    if(this.array.length === 0) {
      throw Error("no elements exist.")
    } else {
      return this.array[this.array.length - 1]
    }
  }

  /**
   * Returns the element at the given index. 
   * @param {number} the index.
   * @returns {T}
   */
  public element(index: number): T {
    if(index < 0) {
      throw Error("index must be greater than 0")
    } else if (index >= this.array.length) {
      throw Error("index is outside the bounds of this sequence.")
    } else {
      return this.array[index]
    }
  }

  /**
   * Skips the given number of elements in the sequence.
   * @param {number} the number of elements to skip.
   * @returns {Sequence<T>}
   */
  public skip(count: number): Sequence<T> {
    let buffer = new Array<T>(this.array.length - count)
    let index  = 0
    for(let i = count; i < this.array.length; i++) {
      buffer[index] = this.array[i]
      index += 1
    } return new Sequence<T>(buffer)
  }

  /**
   * Takes the given number of elements from this sequence. If
   * the take size exceeds the number of elements in the sequence,
   * the take size is reduced to match the number of elements 
   * available.
   * @param {number} the number of elements to take.
   * @returns {Sequence<T>}
   */
  public take(count: number): Sequence<T> {
    if(count > this.array.length) count = this.array.length
    let buffer = new Array<T>(count)
    for(let i = 0; i < count; i++) {
      buffer[i] = this.array[i]
    } return new Sequence<T>(buffer)
  }

  /**
   * reverses this sequence.
   * @returns {Sequence<T>}
   */
  public reverse(): Sequence<T> {
    let buffer = new Array<T>(this.array.length)
    let index = 0
    for(let i = this.array.length - 1; i >= 0; i--) {
      buffer[index] = this.array[i]
      index += 1
    } return new Sequence<T>(buffer)
  }

  /**
   * Creates a sliding window across the elements of this sequence.
   * Each element in the resulting sequence will be an array of the 
   * given size. The step size is the stride the window should take
   * across the sequences. 
   * @param {number} the size of the window.
   * @param {number} the step size.
   * @returns {Sequence<Array<T>>}
   */
  public window(size: number, step: number): Sequence<Array<T>> {
    let buffer = new Array<Array<T>>()
    for(let i = 0; i < this.array.length; i += step) {
      let window = new Array<T>()
      for(let j = 0; j < size; j++) {
        let index = (i + j)
        if(index < this.array.length) {
          window.push(this.array[index])
        }
      } buffer.push(window)
    } return new Sequence<Array<T>>(buffer)
  }

  /**
   * Returns this sequences internal array.
   * @returns {Array<T>}
   */
  public collect(): Array<T> {
    return this.array
  }

  /**
   * Creates a linear numeric Sequence sequence with the given range.
   * @param {number} the starting index.
   * @param {number} the ending value.
   */
  public static range(from: number, to: number): Sequence<number> {
    let buffer = new Array<number>(to - from)
    let index = 0
    for(let i = from; i < to; i++) {
      buffer[index] = i
      index += 1
    } return new Sequence<number>(buffer)
  }

  /**
   * Creates a new sequence from the given array.
   * @param {Array<T>} the array.
   * @returns {Sequence<T>}
   */
  public static fromArray<T>(array: Array<T>): Sequence<T> {
    return new Sequence(array)
  }
}