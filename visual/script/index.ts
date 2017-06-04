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

import * as neuron from "../../src/index"
import { ready } from "./ready"
import { loop } from "./loop"
import { Device } from "./device"

ready(() => {

  //-----------------------------------------
  // initialize elements
  //-----------------------------------------
  const info = {
    iteration: document.querySelector("#iteration"),
    error: document.querySelector("#error"),
    src: document.querySelector("#src")
  }
  const src = new Device("#src", {
    resolutionX: 32,
    resolutionY: 32,
    width: 256,
    height: 256
  })
  const dst = new Device("#dst", {
    resolutionX: 32,
    resolutionY: 32,
    width: 256,
    height: 256
  })
  
  //-----------------------------------------
  // reinitialize network.
  //-----------------------------------------
  info.src.addEventListener("click", () => {
    src.clear(0)
    for (let i = 0; i < 10; i++) {
      const ox = Math.floor(Math.random() * src.resX())
      const oy = Math.floor(Math.random() * src.resY())
      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
          src.set(ox + x, oy + y, 1)
        }
      }
    }
  })

  //-----------------------------------------
  // initialize network
  //-----------------------------------------
  const network = new neuron.Trainer(new neuron.Network([
    new neuron.Tensor(2),
    new neuron.Tensor(8, "tanh"),
    new neuron.Tensor(8, "tanh"),
    new neuron.Tensor(8, "tanh"),
    new neuron.Tensor(1, "tanh"),
  ]), {
      step: 0.015
    })


  //-----------------------------------------
  // initialize random image.. (5x5 plots)
  //-----------------------------------------
  for (let i = 0; i < 10; i++) {
    const ox = Math.floor(Math.random() * src.resX())
    const oy = Math.floor(Math.random() * src.resY())
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        src.set(ox + x, oy + y, 1)
      }
    }
  }

  let iteration = 0
  loop(() => {
    //-----------------------------------------
    // feed src image to network.
    //-----------------------------------------
    let error = 0
    for (let i = 0; i < 1; i++) {
      error = 0
      for (let y = 0; y < src.resY(); y++) {
        for (let x = 0; x < src.resX(); x++) {
          error += network.backward([x / src.resX(), y / src.resY()], [src.get(x, y)])
        }
      }
    }

    //-----------------------------------------
    // show network output.
    //-----------------------------------------
    for (let y = 0; y < src.resX(); y++) {
      for (let x = 0; x < src.resY(); x++) {
        dst.set(x, y, network.forward([x / 32, y / 32])[0])
      }
    }
    //-----------------------------------------
    // update info
    //-----------------------------------------
    info.error.innerHTML = (error / (src.resX() * src.resY())).toString()
    info.iteration.innerHTML = iteration.toString()

    iteration += 1
  })
})
