"""Generate dense embeddings. """

import os 
import math 
import hydra
import pickle 
import pathlib
import logging 

import torch 
import torch.nn as nn 

from typing import List, Tuple 
from omegaconf import DictConfig, OmegaConf
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
from dpr.models import init_biencoder_components
from dpr.data.biencoder_data import BiEncoderTable
from dpr.data.table_data import prepare_table_ctx_inputs_batch


logger = logging.getLogger()
setup_logger(logger)



def get_table_ctx_vectors(
    cfg: DictConfig, 
    ctx_rows: List[Tuple[object, BiEncoderTable]], 
    model: nn.Module, 
    tensorizer: Tensorizer, 
    insert_title: bool = True, 
): 
    """Encode table with context encoder under global/rowcol/auxemb settings."""
    n = len(ctx_rows) 
    bsz = cfg.batch_size 
    total = 0 
    results = [] 
    for j, batch_start in enumerate(range(0, n, bsz)): 
        batch = ctx_rows[batch_start: batch_start + bsz] 
        input_tensors = prepare_table_ctx_inputs_batch(
            batch, 
            tensorizer.tokenizer, 
            cfg.structure_option, 
            insert_title, 
            cfg.max_sequence_length, 
        )
        ctx_ids_batch = move_to_device(input_tensors['token_ids'], cfg.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch).long(), cfg.device)
        
        if cfg.structure_option == "rowcol": 
            ctx_attn_mask = move_to_device(input_tensors['attn_mask'], cfg.device) 
        else: 
            ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), cfg.device)
        
        if cfg.structure_option == 'auxemb': 
            ctx_row_batch = move_to_device(input_tensors['row_ids'], cfg.device) 
            ctx_col_batch = move_to_device(input_tensors['column_ids'], cfg.device)
            with torch.no_grad(): 
                _, out, _ = model(
                    input_ids=ctx_ids_batch, 
                    token_type_ids=ctx_seg_batch, 
                    attention_mask=ctx_attn_mask, 
                    row_ids=ctx_row_batch, 
                    column_ids=ctx_col_batch, 
                )
        elif cfg.structure_option == "biased": 
            ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), cfg.device)
            ctx_bias_mask_id = move_to_device(input_tensors['row_ids'], cfg.device) 
            ctx_col_batch = move_to_device(input_tensors['column_ids'], cfg.device)
            with torch.no_grad(): 
                _, out, _ = model(
                    input_ids=ctx_ids_batch, 
                    token_type_ids=ctx_seg_batch, 
                    attention_mask=ctx_attn_mask, 
                    row_ids=ctx_bias_mask_id, 
                    column_ids=ctx_col_batch, 
                )
        else: 
            with torch.no_grad(): 
                _, out, _ = model(
                    input_ids=ctx_ids_batch, 
                    token_type_ids=ctx_seg_batch, 
                    attention_mask=ctx_attn_mask, 
                )
        out = out.cpu() 
        
        ctx_ids= [r[0] for r in batch] 
        extra_info = [] 
        if len(batch[0]) > 3: extra_info = [r[3:] for r in batch]
        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)
        
        if extra_info:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i]) for i in range(out.size(0))])
        else:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

        if total % 10 == 0: logger.info("Encoded passages %d", total)
            
    return results 
        



@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):

    assert cfg.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg.ctx_src, "Please specify passages source as ctx_src param"

    cfg = setup_cfg_gpu(cfg)

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder, None, cfg.device, cfg.n_gpu,
        cfg.local_rank, cfg.fp16, cfg.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state, strict=False)

    # load from table data sources 
    logger.info("reading data source: %s", cfg.ctx_src)

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict, cfg)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    shard_size = math.ceil(len(all_passages) / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx, end_idx, len(all_passages),
    )
    shard_passages = all_passages[start_idx:end_idx]
    
    data = get_table_ctx_vectors(cfg, shard_passages, encoder, tensorizer, insert_title=True)

    file = cfg.out_file + "_" + str(cfg.shard_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)



if __name__ == "__main__":
    main()