# Tuning the model using curated samples under different settings 

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi

DATA="datasets"
CTX="nq_table" 
CONF="biencoder_nq"   # ("biencoder_nq" "biencoder_local" "biencoder_default")
opt="global"          # ("global" "rowcol" "auxemb" "biased")  

TRAIN_DATA=${ROOT}/${DATA}/${CTX}/"train.converted" 
DEV_DATA=${ROOT}/${DATA}/${CTX}/"dev.converted"

MODEL=${ROOT}/"checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp"

cd ${ROOT}

# 'glocal' OR 'rowcol'
python train_biencoder.py \
  model_file=${MODEL} \
  train=${CONF} \
  train_datasets=[${TRAIN_DATA}] \
  dev_datasets=[${DEV_DATA}] \
  output_dir=${ROOT}/"checkpoint" \
  encoder.encoder_model_type="hf_bert" 
  checkpoint_file_name=${CTX}"_"${opt} \
  ignore_checkpoint_offset=True \
  ignore_checkpoint_lr=True \
  structure_option=${opt}


# # 'auxemb'
# python train_biencoder.py \
#   model_file=${MODEL} \
#   train=${CONF} \
#   train_datasets=[${TRAIN_DATA}] \
#   dev_datasets=[${DEV_DATA}] \
#   output_dir=${ROOT}/"checkpoint" \
#   encoder.encoder_model_type="hf_bert_mix" 
#   checkpoint_file_name=${CTX}"_"${opt} \
#   ignore_checkpoint_optimizer=True \
#   ignore_checkpoint_offset=True \
#   ignore_checkpoint_lr=True \
#   structure_option=${opt}

# # 'biased'
# python train_biencoder.py \
#   model_file=${MODEL} \
#   train=${CONF} \
#   train_datasets=[${TRAIN_DATA}] \
#   dev_datasets=[${DEV_DATA}] \
#   output_dir=${ROOT}/"checkpoint" \
#   encoder.encoder_model_type="hf_bert_bias" 
#   checkpoint_file_name=${CTX}"_"${opt} \
#   ignore_checkpoint_optimizer=True \
#   ignore_checkpoint_offset=True \
#   ignore_checkpoint_lr=True \
#   structure_option=${opt}