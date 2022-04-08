"""Table data processing functions. """

from re import A
import torch 
from typing import Dict, List, Tuple
from .biencoder_data import BiEncoderTable

import logging 
logger = logging.getLogger(__name__)

import transformers
if transformers.__version__.startswith("4"): from transformers import BertTokenizer
else: from transformers.tokenization_bert import BertTokenizer


def to_max_len(seq_list: List[int], pad_id: int, max_len: int): 
    if len(seq_list) < max_len: 
        seq_list.extend([pad_id for _ in range(max_len-len(seq_list))])
    seq_list = seq_list[: max_len]
    return seq_list 



def create_global_attn_mask(max_seq_length: int) -> torch.LongTensor: 
    mask = torch.ones(max_seq_length, max_seq_length).long() 
    return mask 
    

def create_rowcol_attn_mask(
    row_ids: List[int], 
    column_ids: List[int], 
    title_len: int, 
) -> torch.LongTensor:
    # tokens within the same row OR column are mutually visible 
    mask = (row_ids == row_ids.transpose(0, 1)) & (column_ids == column_ids.transpose(0, 1)) 
    mask = mask.long() 
    # title tokens are globally visible 
    mask[: title_len, :] = 1 
    mask[:, : title_len] = 1 
    return mask 
    


def prepare_table_ctx_inputs(
    ctx: BiEncoderTable, 
    tokenizer: BertTokenizer, 
    structure_option: str = "global", 
    insert_title: bool = True, 
    max_seq_length: int = 256, 
): 
    """Tokenize a single table. """
    token_ids = []
    row_ids, col_ids = [], [] 
    
    if insert_title: 
        if hasattr(ctx, "title"): 
            text = ctx.title.strip() 
        else: 
            text = ctx['title'].strip()
        title_token_ids = tokenizer.encode(
            text, 
            add_special_tokens=True, 
            max_length=max_seq_length, 
            truncation=True, 
            pad_to_max_length=False
        )
        title_token_ids = title_token_ids[1: ]  # remove [CLS]
        title_len = len(title_token_ids)
        token_ids.extend(title_token_ids)
        row_ids.extend([0 for _ in title_token_ids])
        col_ids.extend([0 for _ in title_token_ids])
    
    if hasattr(ctx, "cells"): 
        cell_list = ctx.cells
    else: 
        cell_list = ctx['cells']
    for cell in cell_list: 
        text = cell['text'].strip()
        cell_token_ids = tokenizer.encode(
            text, 
            add_special_tokens=False, 
            max_length=max_seq_length, 
            truncation=True, 
            pad_to_max_length=False
        ) 
        token_ids.extend(cell_token_ids) 
        row_ids.extend([cell['row_idx'] for _ in cell_token_ids]) 
        col_ids.extend([cell['col_idx'] for _ in cell_token_ids]) 
    
    assert len(token_ids) == len(row_ids) == len(col_ids) 
    valid_len = min(len(token_ids), max_seq_length)

    token_ids = to_max_len(token_ids, tokenizer.pad_token_id, max_seq_length)
    token_ids[-1] = tokenizer.sep_token_id
    token_ids = torch.LongTensor(token_ids)
    row_ids = torch.LongTensor(to_max_len(row_ids, 0, max_seq_length)) 
    col_ids = torch.LongTensor(to_max_len(col_ids, 0, max_seq_length))   # [max-len]

    if structure_option == "rowcol": 
        attn_mask = create_rowcol_attn_mask(row_ids.unsqueeze(0), col_ids.unsqueeze(0), title_len)
    else: 
        attn_mask = create_global_attn_mask(max_seq_length)

    # set pad positions to invisible 
    attn_mask[valid_len: , :] = 0 
    attn_mask[:, valid_len: ] = 0 
    
    if structure_option == "biased": 
        bias_mask_id = create_biased_id(row_ids, col_ids) 
        return {
            'token_ids': token_ids, 
            'attn_mask': attn_mask, 
            'row_ids': bias_mask_id, 
            'column_ids': col_ids, 
        }

    return {
        'token_ids': token_ids, 
        'attn_mask': attn_mask, 
        'row_ids': row_ids, 
        'column_ids': col_ids, 
    }



def prepare_table_ctx_inputs_batch(
    batch: List[Tuple[object, BiEncoderTable]], 
    tokenizer: BertTokenizer, 
    structure_option: str = "global", 
    insert_title: bool = True, 
    max_seq_length: int = 256, 
) -> Dict[str, torch.Tensor]: 
    token_ids_batch, attn_mask_batch = [], [] 
    row_ids_batch, column_ids_batch = [], []
    for ctx in batch: 
        ctx_input_tensors = prepare_table_ctx_inputs(
            ctx[1], tokenizer, structure_option, 
            insert_title, max_seq_length, 
        )
        token_ids_batch.append(ctx_input_tensors['token_ids'])
        attn_mask_batch.append(ctx_input_tensors['attn_mask'])
        row_ids_batch.append(ctx_input_tensors['row_ids'])
        column_ids_batch.append(ctx_input_tensors['column_ids']) 
    
    token_ids_batch = torch.stack(token_ids_batch, dim=0)   # [batch-size, max-seq-len]
    attn_mask_batch = torch.stack(attn_mask_batch, dim=0)   # [batch-size, max-seq-len, max-seq-len]
    row_ids_batch = torch.stack(row_ids_batch, dim=0)       # [batch-size, max-seq-len]
    column_ids_batch = torch.stack(column_ids_batch, dim=0) # [batch-size, max-seq-len]
    
    return {
        'token_ids': token_ids_batch, 
        'attn_mask': attn_mask_batch, 
        'row_ids': row_ids_batch, 
        'column_ids': column_ids_batch, 
    }

# %% biased 

def create_biased_id(row_ids: torch.Tensor, column_ids: torch.Tensor) -> torch.Tensor: 
    """Compute relation-based bias id. 
    args: 
        row_ids: <seq-len> 
        column_ids: <seq-len> 
        valid_len: <int> 
    ret: 
        bias_id: <seq-len, seq-len> 
        
    notes: 
     - title:  row-id = 0, col-id = 0 
     - header: row-id = 0, col-id = 1-indexed 
     - cell:   row-id = 1-indexed, col-id = 1-indexed 
    """
    n = row_ids.size()[0]
    bias_id = [] 
    
    for i in range(n): 
        i_bid = [] 
        irow, icol = row_ids[i], column_ids[i] 
        for j in range(n): 
            jrow, jcol = row_ids[j], column_ids[j] 
            
            if (irow == 0) and (icol == 0):   # [f] sentence
                if (jrow == 0) and (jcol == 0):         # [t] sentence 
                    ij_bid = 0 
                elif (jrow == 0):                       # [t] header 
                    ij_bid = 1 
                else:                                   # [t] cell 
                    ij_bid = 2 
            elif (irow == 0):                 # [f] header 
                if (jrow == 0) and (jcol == 0):         # [t] sentence 
                    ij_bid = 3 
                elif (jrow == 0) and (icol == jcol):    # [t] same header
                    ij_bid = 4 
                elif (jrow == 0):                       # [t] other header
                    ij_bid = 5 
                else:                                   # [t] cell 
                    ij_bid = 6 
            else:                             # [f] cell 
                if (jrow == 0) and (jcol == 0):         # [t] sentence 
                    ij_bid = 7 
                elif (jrow == 0):                       # [t] column header 
                    ij_bid = 8 
                elif (irow == jrow) and (icol == jcol): # [t] same cell 
                    ij_bid = 9 
                elif (irow == jrow):                    # [t] same row 
                    ij_bid = 10 
                elif (icol == jcol):                    # [t] same column 
                    ij_bid = 11 
                else: 
                    ij_bid = 12 
            
            i_bid.append(ij_bid)
    
        bias_id.append(i_bid)
    return torch.LongTensor(bias_id)
