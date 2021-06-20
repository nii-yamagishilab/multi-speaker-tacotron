# Scripts for creating source and target tfrecords for synthesis from plain text

## Dependencies

The text-to-phoneme conversion is done using Flite.  Please get a copy of Flite from here:  https://github.com/festvox/flite

## Environment

Create the preprocess-cpu environment.

```conda create -n preprocess-cpu python=3.6.8
conda activate preprocess-cpu
conda install unidecode pyspark docopt
conda install -c conda-forge librosa
pip install inflect
conda install tensorflow=1.11
```

## Modify paths in scripts

In `construct_tfrecords.py`, modify the paths and speaker ID variables to match your environment and data.

## Run

```conda activate preprocess-cpu
python construct_tfrecords.py```

This outputs source and target tfrecords.  The target tfrecords are just empty audio but they need to be created because Tacotron expects both source and target data even at synthesis time.

The example texts come from the CMU ARCTIC sentences  http://www.festvox.org/cmu_arctic/

The hparams.json in this directory are the same ones that were used in training the VCTK Tacotron models.
