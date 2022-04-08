# Curate the training and validation samples for NQ-Table. 

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi

declare -a split_list=("dev" "train") 

DATA="datasets"
CTX="nq_table"

EMBED=${ROOT}/${DATA}/${ctx}/"embed"
TABLES=${ROOT}/${DATA}/${ctx}/"tables_proc.jsonl"

cd ${ROOT}

for spt in ${split_list[@]}; do

    # [0] declare output files 
    RETRIEVED=${ROOT}/${DATA}/${CTX}/${spt}".retrieved" 
    CONVERTED=${ROOT}/${DATA}/${CTX}/${spt}".converted" 
    ANNOTATED=${ROOT}/${DATA}/${CTX}/${spt}".jsonl"

    # [1] run retrieval inference using the generated embeddings 
    echo "[1] run ["${CTX}"] retrieval inference"
    python dense_retrieval.py \
        ctx_datatsets=[${CTX}"_all"] \
        encoded_ctx_files=[${EMBED}"_0"] \
        qa_dataset=${CTX}"_"${spt} \
        out_file=${RETRIEVED} 
    
    # [2] convert to training format 
    echo "[2] ["${CTX}"] convert "${spt}" results"
    python convert_data.py \
        --tables_file=${TABLES} \
        --retrieved_path=${RETRIEVED} \
        --converted_path=${CONVERTED} \
        --annotated_path=${ANNOTATED}

done 