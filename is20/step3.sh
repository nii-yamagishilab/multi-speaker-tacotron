#!/bin/sh
#$ -cwd
#$ -l s_gpu=1
#$ -l h_rt=24:00:00
# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper (ecooper@nii.ac.jp)
# All rights reserved.
# ==============================================================================

## VCTK+ASR model with channel codes -> add dialect embeddings

. /etc/profile.d/modules.sh
module load intel cuda/9.0.176 nccl/2.2.13 cudnn/7.3

export PATH="/home/7/18IA1182/miniconda3/bin:$PATH"
source activate tacotron2

export PYTHONPATH=/home/7/18IA1182/external:/home/7/18IA1182/external/tacotron2:/home/7/18IA1182/external/self_attention_tacotron:/home/7/18IA1182/external/multi_speaker_tacotron:$PYTHONPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true

cd /home/7/18IA1182/external/self_attention_tacotron

INTYPE=phone
NAME=vctk_spk_resnet_mfcc_3-8_256_64_mean+std_lde_sqr_softmax
GEN=F
LIDDIM=256
CKPT=199485

python train.py --source-data-root=/gs/hs0/tgh-20IAA/ecooper/data/interspeech/source --target-data-root=/gs/hs0/tgh-20IAA/ecooper/data/interspeech/target --checkpoint-dir=/gs/hs0/tgh-20IAA/ecooper/experiments/test_is_code/checkpoint-step3 --selected-list-dir=/gs/hs0/tgh-20IAA/ecooper/data/interspeech/ttd/$GEN --hparams=tacotron_model="DualSourceSelfAttentionTacotronModel",encoder="SelfAttentionCBHGEncoder",decoder="DualSourceTransformerDecoder",initial_learning_rate=0.00005,decay_learning_rate=False,cbhg_out_units=512,use_accent_type=False,embedding_dim=512,encoder_prenet_out_units=[512,512],encoder_prenet_drop_rate=0.5,projection1_out_channels=512,projection2_out_channels=512,self_attention_out_units=64,self_attention_encoder_out_units=64,decoder_prenet_out_units=[256,256],decoder_out_units=1024,attention_out_units=128,attention1_out_units=128,attention2_out_units=64,decoder_self_attention_num_hop=2,decoder_self_attention_out_units=1024,outputs_per_step=2,max_iters=500,attention=forward,attention2=additive,cumulative_weights=False,attention_kernel=31,attention_filters=32,use_zoneout_at_encoder=True,decoder_version="v2",num_symbols=256,eval_throttle_secs=600,eval_start_delay_secs=120,num_evaluation_steps=200,keep_checkpoint_max=200,use_l2_regularization=True,l2_regularization_weight=1e-7,use_postnet_v2=True,batch_size=32,dataset="vctk.dataset.DatasetSource",save_checkpoints_steps=1683,target_file_extension="target.tfrecord",use_external_speaker_embedding=True,embedding_file="/gs/hs0/tgh-20IAA/ecooper/data/interspeech/vctk-lde-3.txt",speaker_embedding_dim='512',speaker_embedding_projection_out_dim=64,speaker_embedd_to_decoder=True,num_speakers=719,speaker_embedding_offset=5,channel_id_to_postnet=True,channel_id_file="/gs/hs0/tgh-20IAA/ecooper/data/interspeech/channel_encodings/corpus_channel_encodings.txt",channel_id_dim=5,use_language_embedding=True,language_embedding_projection_out_dim=64,language_embedding_file=/gs/hs0/tgh-20IAA/ecooper/data/interspeech/jeff_LIDs/$INTYPE/$NAME.txt,language_embedding_dim=$LIDDIM,language_embedd_to_input=True,language_embedd_to_decoder=False,source=$INTYPE,warm_start=True,ckpt_to_initialize_from=/gs/hs0/tgh-20IAA/ecooper/experiments/test_is_code/checkpoint-step2,vars_to_warm_start=["^((?!dense_1/kernel|dense_1/bias|dense_2/kernel|dense_2/bias).)*$"],logfile=/gs/hs0/tgh-20IAA/ecooper/experiments/test_is_code/step3.log --hparam-json-file=/gs/hs0/tgh-20IAA/ecooper/data/interspeech/target/hparams.json
