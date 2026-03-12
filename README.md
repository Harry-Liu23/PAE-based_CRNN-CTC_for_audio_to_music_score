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



## Citation

If you use this code, please cite the original paper:

```bibtex
@article{roman2020audio2score,
  title={Data representations for audio-to-score monophonic music transcription},
  author={Román, Miguel A. and Pertusa, Antonio and Calvo-Zaragoza, Jorge},
  journal={Expert Systems with Applications},
  year={2020}
}
