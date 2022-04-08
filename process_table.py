"""Preprocess the NQ-Table. 
dst: {'id': str, 'title': str, 'cells': ['text': str, 'row_idx': 0, 'col_idx': 0]}
"""

import sys 
import json 
import random 

from typing import Dict 

from dpr.data.biencoder_data import get_processed_table_dict


# %% ablation: shuffling the content, by a given extent `prob`

def shuffle_nq_table_by_row(table: Dict, prob: float = 0.5): 
    num_rows = len(table['rows'])
    for i in range(num_rows): 
        p = random.random() 
        if (p < prob): 
            random.shuffle(table['rows'][i]['cells'])
    return table 


def shuffle_nq_table_by_column(table: Dict, prob: float = 0.5): 
    num_rows = len(table['rows'])
    
    column_sizes = [len(row['cells']) for row in table['rows']]
    num_columns = min(column_sizes)
    
    for j in range(num_columns): 
        p = random.random()
        if (p < prob): 
            # collect cells in the j-th column 
            cells = [row['cells'][j] for row in table['rows']]
            random.shuffle(cells) 
            for i in range(num_rows): 
                table['rows'][i]['cells'][j] = cells[i] 
    return table 


def preprocess_nq_tables_perturbed(
    original_tables_path: str, 
    processed_tables_path: str, 
    mode: int,    # 0: by row, 1: by column, 2: both
    prob: float = 0.5, 
): 
    fr = open(original_tables_path, 'r') 
    fw = open(processed_tables_path, 'w') 
    
    for idx, orig_line in enumerate(fr): 
        # if (idx > 10): break
        orig_table = json.loads(orig_line.strip()) 
        if (mode % 2) == 0: 
            orig_table = shuffle_nq_table_by_row(orig_table, prob)
        if mode == 1: 
            orig_table = shuffle_nq_table_by_column(orig_table, prob)
         
        proc_table = get_processed_table_dict(
            orig_table, 
            row_selection='none',
            max_cell_num=120, 
            max_words=120, 
            max_words_per_header=10, 
            max_words_per_cell=8, 
            max_cell_num_per_row=64, 
            header_delimiter='|', 
            cell_delimiter='|', 
            row_delimiter='.', 
            return_dict=True, 
    )
        proc_line = json.dumps(proc_table) 
        fw.write(f"{proc_line}\n")
    
    fr.close()
    fw.close() 



# %% normal pre-process
# ablation: header/cell and row delimiters 

def preprocess_nq_tables(
    original_tables_path: str, 
    processed_tables_path: str, 
    keep_all: bool = False, 
    cell_delim: bool = True, 
    row_delim: bool = True, 
): 
    if cell_delim: 
        cell_delimiter = '|'
        header_delimiter = '|'
    else: 
        cell_delimiter = ''
        header_delimiter = ''
    
    if row_delim: 
        row_delimiter = '.'
    else: 
        row_delimiter = ''
    
    fr = open(original_tables_path, 'r') 
    fw = open(processed_tables_path, 'w') 
    
    for idx, orig_line in enumerate(fr): 
        # if (idx > 10): break
        orig_table = json.loads(orig_line.strip()) 
        if keep_all: 
            proc_table = get_processed_table_dict(
                orig_table, 
                row_selection='none',
                max_cell_num=100000, 
                max_words=100000, 
                max_words_per_header=100, 
                max_words_per_cell=100, 
                max_cell_num_per_row=1000, 
                header_delimiter='', 
                cell_delimiter='', 
                row_delimiter='', 
                return_dict=True, 
            )
        else: 
            proc_table = get_processed_table_dict(
                orig_table, 
                row_selection='random',
                max_cell_num=None, 
                max_words=112, 
                max_words_per_header=12, 
                max_words_per_cell=8, 
                max_cell_num_per_row=64, 
                header_delimiter=header_delimiter, 
                cell_delimiter=cell_delimiter, 
                row_delimiter=row_delimiter, 
                return_dict=True, 
            )
        proc_line = json.dumps(proc_table) 
        fw.write(f"{proc_line}\n")
    
    fr.close()
    fw.close() 



# %% main     

if __name__ == "__main__": 
    
    exp = sys.argv[1]
    
    if exp == "nq":   # 1. nq-table 
        preprocess_nq_tables(
            original_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables.jsonl", 
            processed_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables_all.jsonl", 
            keep_all=True, 
        ) 
        preprocess_nq_tables(
            original_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables.jsonl", 
            processed_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables_proc_2.jsonl", 
            keep_all=False, 
        ) 
    elif exp == 'perturb':  # 2. ablation (perturb)
        preprocess_nq_tables_perturbed(
            original_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables.jsonl", 
            processed_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables_row.jsonl", 
            mode=0, 
            prob=1.0, 
        ) 
        preprocess_nq_tables_perturbed(
            original_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables.jsonl", 
            processed_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables_column.jsonl", 
            mode=1, 
            prob=1.0, 
        ) 
        preprocess_nq_tables_perturbed(
            original_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables.jsonl", 
            processed_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables_both.jsonl", 
            mode=2, 
            prob=1.0, 
        )
    elif exp == 'delim':  # 3. ablation (delimiter)
        preprocess_nq_tables(
            original_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables.jsonl", 
            processed_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables_dcell.jsonl", 
            keep_all=False, 
            cell_delim=True, 
            row_delim=False, 
        ) 
        preprocess_nq_tables(
            original_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables.jsonl", 
            processed_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables_drow.jsonl", 
            keep_all=False, 
            cell_delim=False, 
            row_delim=True,
        ) 
        preprocess_nq_tables(
            original_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables.jsonl", 
            processed_tables_path="/mnt/zhiruow/hitab/table-retrieval/datasets/nq_table/tables_dnone.jsonl", 
            keep_all=False, 
            cell_delim=False, 
            row_delim=False,
        ) 
