# Open-Domain Table Retrieval for Natural Questions

This repository involves the data and code for the paper:

[Table Retrieval May Not Necessitate Table-specific Model Design]()


## Preliminaries 
To install the necessary libraries, run 
```
pip install . 
```
You'll also need the pre-trained DPR model checkpoint for (1) evaluating its zero-shot performance, and (2) initialize the model instance before start the fine-tuning. 
To do this, run 
```
cd ./dpr/ 

python data/download_data.py \
  --resource checkpoint.retriever.single-adv-hn.nq.bert-base-encoder \
  --output_dir ../../downloads/checkpoint
```

remember to set the ROOT before running the given scripts: 
```
export ROOT_DIR=`pwd`
```


## Zero-shot Retrieval Inference 
To perform zero-shot retrieval inference on the NQ-table table retrieval dataset, we need to first generate the embeddings of all tables (`generate_embeddings.py`), then, encode question/queries in-time and search for the most relevant tables (`dense_retrieval.py`). 

To automate the entire pipeline, just execute the `scripts/run_inference.sh`. This will iterate each dataset to generate context embeddings and run inference accordingly. 

If you want a more concrete walk-thru of each module, we detail the them as follows: 

**Generate Context Embeddings**

Different table contexts, located in different files and may need to be loaded with different classes, are specified in the `conf/ctx_sources/table_sources.yaml` file. One can alter the `ctx_src` argument when calling the `generate_embeddings.py` script. 
By default, we use NQ-Table which is denoted as `nq_table`. 

For example, to generate embeddings of NQ-Table, run: 
```
python generate_embeddings.py 
  ctx_src=nq_table \
  out_file=${your_path_to_store_embeddings} 
```


**Retrieve Relevant Tables for Questions/Queries** 

In this step, we need to specify the file(s) containing questions so as to pair relevant tables for them using the generated embeddings. 
Likewise, these files are included in the `conf/datasets/table_retrieval.yaml` file. One can alter the `qa_dataset` argument to load different questions, when calling the `dense_retrieval.py`. 

The train/dev/test sets of NQ-Table are indicated by `nq_table_train`/`nq_table_dev`/`nq_table_test`. 

For example, to run retrieval inference on NQ-Table test questions, run: 
```
python dense_retrieval.py 
  ctx_datatsets=[nq_table] \
  encoded_ctx_files=[${your_path_to_store_embeddings}"_0"] \
  qa_dataset=nq_table_test \
  out_file=${your_path_for_the_retrieval_result.json} 
```


## Fine-tune with Model Variants 

**Settings**

Neither DPR has table-specific designs nor has it been trained on tables. 
We further explore the benefit of (1) augmented fine-tuning, and (2) add auxiliary structure-aware modules. 

The first and naive version of fine-tuning feeds models with serialized table content and applies **no** model modifcations. We denote this as the `global` setting (since it applies a global attention against structurally restricted ones). 

The other three fine-tune setting adppts the two major methods to incorporate table structures. 
1. Adding auxiliary embeddings, specifically for row and column indices. We denote this as the `auxemb` setting. 
2. Applying structure-aware attention, by enforcing tokens to be visible in-row or in-column. This is denoted as the `rowcol` setting. 
3. Adding relation-based attention bias onto the global self-attention scores, denoted as `biased`. 

Among experiments, one can alter between these three settings by specifiing the `structure_option` argument. 
For `auxemb` and `biased` which requires extra parameters (hence change in model architecure), alter the encoder type by additionally specifying
```
encoder.encoder_model_type="hf_bert_mix"   # or "hf_bert_bias"
```


**Creating Training (and Validation) Dataset**

To obtain the most effective training data, we follow the hard-negative selection strategy and leverage the retrieval results for sample curation. To be more concrete, for trainable datasets (NQ-Table and WebQueryTable), we firstly run zero-shot retrieval for training and validation samples. Then for each question and its retrieved 100 table contexts, we categorize them into (1) positive, (2) negative, (3) hard negative. To (1) if it contains the answer text, and to (2)/(3) otherwise. If the context ranks among the top-20, it goes into (3), otherwise would be a rather simple negative context and goes into (2). 

To implement this, we also need to run `dense_retrieval.py` inference using the generated table context embeddings. 
Then, convert the retrieval result into training format using `convert_data.py`. 
```
python get_trainset_from_retrieved.py \
  ${raw_tables_path} \
  ${retrieved_result} \
  ${converted_training_data}  
```
Remember to do this for both of your training and validation samples. 

One can also automate this process by running the `scripts/curate_data.sh`


**Bi-Encoder Training**

With the curated datasets, we can then start fine-tuning using `train_biencoder.py`. Viable training options reads in the `conf/datasets/biencoder_train.yaml`. 

```
python train_biencoder.py \
  train=biencoder_nq \
  train_datasets=[${your_train_data_pile}] \
  dev_datasets=[${your_dev_data_file}] \
  output_dir=${directory_for_checkpoints} \
  checkpoint_file_name=${model_name}
```
Alter the arguments in `conf/biencoder_train.yaml` to train in different experimental setings. 

Or simply, just run `scripts/tune_model.sh`.


**Evaluate with Tuned Models**

Similarly to the zero-shot inference, but probably need to specified the fine-tuned model file, as well as the new names for embedding and retrieval results. See `scripts/run_inference.sh` for more details. 



## Ablation Study (NQ-Table)
**Delimiter**

Use the `process_tables.py` to create table contexts linearized using different delimiters. 
Alter the arguments `header_delimiter`, `cell_delimiter`, and `row_delimiter` to compare. 

**Structure Perturbation**

Use the `process_tables.py` to create processed tables shuffled in different orientations (by row, by column) and to the designated extent (prob default to 0.5). This will create `datasets/nq_table/tables_row.jsonl` and `datasets/nq_table/tables_column.jsonl`. 


## Citation

```
@article{wang2022table,
  title={Table Retrieval May Not Necessitate Table-specific Model Design},
  author={Wang, Zhiruo and Jiang, Zhengbao and Nyberg, Eric and Neubig, Graham},
  journal={arXiv preprint arXiv:2205.09843},
  year={2022}
}
```
