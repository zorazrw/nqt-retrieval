# Zero-shot Inference for NQ-Table using DPR checkpoint 

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi

declare -a split_list=("test") 

DATA="datasets"
CTX="nq_table"

MODEL=${ROOT}/"checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp"
EMBED=${ROOT}/${DATA}/${CTX}/"embed"


cd ${ROOT}

# [1] generate dense embeddings
echo "[1] generating ["${CTX}"] embeddings"
python generate_embeddings.py \
  model_file=${MODEL} \
  ctx_src=${CTX} \
  out_file=${EMBED}

# [2] run retrieval inference using the generated embeddings 
echo "[2] run ["${CTX}"] retrieval inference"

for spt in ${split_list[@]}; do
  RETRIEVED=${ROOT}/${DATA}/${CTX}/${spt}".retrieved"

  python dense_retrieval.py \
    model_file=${MODEL} \
    ctx_datatsets=[${CTX}"_all"] \
    encoded_ctx_files=[${EMBED}"_0"] \
    qa_dataset=${CTX}"_"${spt} \
    out_file=${RETRIEVED}

done 