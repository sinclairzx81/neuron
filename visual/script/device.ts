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

export interface DeviceOptions {
  resolutionX : number
  resolutionY : number
  width       : number
  height      : number
}
export class Device {
  private canvas : HTMLCanvasElement
  private context: CanvasRenderingContext2D
  constructor(private selector: string, private options: DeviceOptions) {
    this.canvas = document.querySelector(selector) as HTMLCanvasElement
    this.canvas.width        = this.options.resolutionX
    this.canvas.height       = this.options.resolutionY
    this.canvas.style.width  = `${this.options.width}px`
    this.canvas.style.height = `${this.options.height}px`
    this.context = this.canvas.getContext("2d")
    for (let y = 0; y < this.resY(); y++) {
      for (let x = 0; x < this.resX(); x++) {
        this.set(x, y, 0)
      }
    }
  }
  /**
   * returns the x resolution for this device.
   * @returns {number}
   */
  public resX(): number {
    return this.options.resolutionX
  }
  /**
   * returns the y resolution for this device.
   * @returns {number}
   */
  public resY(): number {
    return this.options.resolutionY
  }
  /**
   * gets the value at the given x, y postiion
   * @param {number} x the x position.
   * @param {number} y the y position.
   * @returns {number}
   */
  public get(x:number, y: number) : number {
    const imageData = this.context.getImageData(x, y, 1, 1)
    return imageData.data[0] / 256
  }
  /**
   * sets a value at the given x and y position.
   * @param {number} x the x offset.
   * @param {number} y the y offset.
   * @param {number} value the value to set (between 0 and 1)
   * @returns {void}
   */
  public set(x:number, y: number, value: number): void {
    const imageData = this.context.getImageData(x, y, 1, 1)
    value = Math.floor(value * 256)
    imageData.data[0] = value
    imageData.data[1] = value
    imageData.data[2] = value
    imageData.data[3] = 255
    this.context.putImageData(imageData, x, y)
  }
  /**
   * clears the device.
   * @param {number} x the x offset.
   * @param {number} y the y offset.
   * @param {number} value the value to set (between 0 and 1)
   * @returns {void}
   */
  public clear(value: number): void {
    for (let y = 0; y < this.resY(); y++) {
      for (let x = 0; x < this.resX(); x++) {
        this.set(x, y, value)
      }
    }
  }
}