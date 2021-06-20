# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper (ecooper@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import os

##### VARIABLES TO SET YOURSELF #####
speakerID = 's10'  ## should match the ID number you chose for this speaker in your embedding file
## should be of the format [one letter][some digits] -- the first character gets removed and the digits get selected internally
speakerGender = 'M'  ## should match your data
FLITE = '/home/smg/cooper/installs/flite/bin/flite'  ## path to your flite
txtdir = '/home/smg/cooper/make_tfrecords/example_text' ## path to your text files
hparams = '/home/smg/cooper/make_tfrecords/hparams.json'  ## path to your hparams file -- use the one from your training data.  This example is the one that goes with the VCTK training data.
outdir = '/home/smg/cooper/make_tfrecords/test_output'  ## path where you want the output source and target directories to go.
##########

mksourcecmd = 'mkdir -p ' + outdir + '/source'
print(mksourcecmd)
os.system(mksourcecmd)

mktargetcmd = 'mkdir -p ' + outdir + '/target'
print(mktargetcmd)
os.system(mktargetcmd)

source_cmd = "python preprocess.py --source-only --gender=" + speakerGender + " --speakerID=" + speakerID + " --hparams=phoneme=flite,flite_binary_path='" + FLITE + "' " + txtdir + ' ' + outdir + '/source'
print(source_cmd)
os.system(source_cmd)

target_cmd = "python preprocess.py --target-only --gender=" + speakerGender + " --speakerID=" + speakerID + " " + txtdir + " " + outdir + '/target'
print(target_cmd)
os.system(target_cmd)

hparam_cmd = 'cp ' + hparams + ' ' + outdir + '/target/'
print(hparam_cmd)
os.system(hparam_cmd)
