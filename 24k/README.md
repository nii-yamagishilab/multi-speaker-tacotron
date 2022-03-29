# 24kHz Multi-Speaker VCTK Models

Code and models from our technical report on arXiv, "Pretraining Strategies, Waveform Model Choice, and Acoustic Configurations for Multi-Speaker End-to-End Speech Synthesis," by Erica Cooper, Xin Wang, Yi Zhao, Yusuke Yasuda, and Junichi Yamagishi.  https://arxiv.org/abs/2011.04839

Pretrained models can be found in the `24k` directory of the data shared on Zenodo:  https://zenodo.org/record/6349897#.YkKR-C8Rr0o
 * Preprocessed VCTK data: in the `data` directory
 * VCTK Tacotron models: in the `tacotron-models` directory
   * `ljspeech-24k`: single-speaker, character-input model trained on LJSpeech.
   * `vctk-24k`: trained from scratch (phone-based and character-based models)
   * `vctk-libritts-ljspeech`: Single-speaker LJSpeech model is used to warm-start a multi-speaker model from LibriTTS clean-360, which is then fine-tuned using VCTK. (character-based)
   * `vctk-ljspeech-char-24k`: LJSpeech single-speaker model is used to warm-start a multi-speaker VCTK model. (character-based)


## 24kHz configurations

These configurations were chosen in order to use shared settings for both the acoustic model and the waveform model, and also to produce higher-quality 24kHz synthesized speech audio.  Here are the new configurations:
 * 24kHz sampling rate for waveform output
 * 12ms frame shift (288 samples)
 * 50ms frame length (1200 samples)
 * fft size: 2048
 * fft bins: 1025
 * fmin: 0
 * fmax: 8000

fmax is restricted to 8000 instead of 1200 in order to use the same frequency resolution when including datasets that are sampled at 16kHz.


## Licence

BSD 3-Clause License

Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
