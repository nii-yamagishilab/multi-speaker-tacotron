# multi-speaker-tacotron

This is an implementation of our paper to appear at ICASSP 2020:  
"Zero-Shot Multi-Speaker Text-To-Speech with State-of-the-art Neural Speaker Embeddings," by Erica Cooper, Cheng-I Lai, Yusuke Yasuda, Fuming Fang, Xin Wang, Nanxin Chen, and Junichi Yamagishi.  
https://arxiv.org/abs/1910.10838  
Please cite this paper if you use this code.  

## Dependencies:  

It is recommended to set up a miniconda environment for using Tacotron.  https://repo.anaconda.com
```
conda create -n taco python=3.6.8
conda activate taco
conda install tensorflow-gpu scipy matplotlib docopt hypothesis pyspark unidecode
conda install -c conda-forge librosa
pip install inflect pysptk
```

Install this repository
```
git clone https://github.com/nii-yamagishilab/multi-speaker-tacotron-internal.git external/multi_speaker_tacotron
```

Install Tacotron dependencies if you don't have them already:
```
mkdir external
git clone https://github.com/nii-yamagishilab/tacotron2.git external/tacotron2
git clone https://github.com/nii-yamagishilab/self-attention-tacotron.git external/self_attention_tacotron
```
Note the renaming of hyphens to underscores; this is necessary because “-” is an invalid character in Python.

Next, download project data and models, from the dropbox folder <a href=https://www.dropbox.com/sh/rq4lebus0n8tmso/AACldbmKDPRN9YiXrRROjtTSa?dl=0>here</a>:
 * Preprocessed VCTK data: in the `data` directory
 * VCTK Tacotron models: in the `tacotron-models` directory
 * VCTK Wavenet models: in the `wavenet-models` directory
 * Nancy model for parameter initialization: TBA

To use our pre-trained WaveNet models, you will also need our WaveNet implementation which can be found here:
https://github.com/nii-yamagishilab/project-CURRENNT-scripts

## How to use

See the scripts `warmup.sh` (training) and `predictmel.sh` (prediction).  The scripts assume a SLURM-type computing environment.  You will need to change the paths to match your environments and point to your data.  Here are the parameters relevant to multi-speaker TTS:
 * `source-data-root` and `target-data-root`: path to your source and target preprocessed data
 * `selected-list-dir`: train/eval/test set definitions
 * `batch_size`: if you get OOM errors, try reducing the batch size
 * `use_external_speaker_embedding=True`: use speaker embeddings that you provide from a file (see the files in the `speaker_embeddings` directory)
 * `embedding_file`: path to the file containing your speaker embeddings
 * `speaker_embedding_dim`:  dimension should match the dimension in your embedding file <!-- TODO: deprecate this -->
 * `speaker_embedding_projection_out_dim=64`: We found experimentally that projecting the speaker embedding to a lower dimension helped to reduce overfitting.  You can try different values, but to use our pretrained multi-speaker models you will have to use 64.
 * `speaker_embedding_offset`: must match the ID of your first speaker.  <!-- TODO: deprecate this -->

The scripts are set up using `embedding_file="vctk-x-vector.txt",speaker_embedding_dim='200'` which is default x-vectors.  Please change it to `embedding_file="vctk-lde-3.txt",speaker_embedding_dim='512'` to use LDE embeddings from our best system.

<!-- num_speakers does not actually get used with external_embedding so TODO remove this from the scripts. -->

## Acknowledgments

This work was partially supported by a JST CREST Grant (JPMJCR18A6, VoicePersonae project), Japan, and by MEXT KAKENHI Grants (16H06302, 17H04687, 18H04120, 18H04112, 18KT0051, 19K24372), Japan. The numerical calculations were carried out on the TSUBAME 3.0 supercomputer at the Tokyo Institute of Technology.

## Licence

BSD 3-Clause License

Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
