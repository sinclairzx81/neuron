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
 * An simple implementation of a linear congruential generator.
 */
export class Random  {
  private a: number
  private c: number
  private m: number

  /**
   * creates a new linear congruential generator.
   * @param {number} the initial seed value.
   * @returns {Random}
   */
  constructor(private seed?: number) {
    this.seed = this.seed === undefined ? 1 : this.seed
    this.a    = 1103515245
    this.c    = 12345
    this.m    = Math.pow(2, 31)
  } 
  /**
   * returns a random number between 0.0 and 1.0
   * @returns {number} 
   */
  public next(): number {
    this.seed = (this.a * this.seed + this.c) % this.m
    return this.seed / this.m
  }

}