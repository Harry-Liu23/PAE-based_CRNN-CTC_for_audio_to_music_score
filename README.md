# Audio-to-Score Monophonic Music Transcription

This repository provides an implementation of the model proposed in:

> Román, M. A., Pertusa, A., & Calvo-Zaragoza, J. (2020).  
> *Data representations for audio-to-score monophonic music transcription.*  
> *Expert Systems with Applications.*

This implementation is independent and intended for educational and research purposes.

## Overview

The project implements an audio-to-score transcription pipeline for **monophonic music**, based on the architecture described in the paper above.

The original work explores end-to-end transcription from audio into symbolic score representations using a **Convolutional Recurrent Neural Network (CRNN)** trained with **Connectionist Temporal Classification (CTC)**.

This repository focuses on implementing that model and reproducing its core ideas in a practical, readable form.

## Status

This is an **independent implementation**, not the official code from the paper's authors.

The repository is intended for:

- research
- learning
- experimentation
- reproduction of published methods


Copyright (c) 2026 <Harry-Liu23>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



## Citation

If you use this code, please cite the original paper:

```bibtex
@article{roman2020audio2score,
  title={Data representations for audio-to-score monophonic music transcription},
  author={Román, Miguel A. and Pertusa, Antonio and Calvo-Zaragoza, Jorge},
  journal={Expert Systems with Applications},
  year={2020}
}
