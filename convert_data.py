"""Convert the table format in the retrieved ctxs from text to table dict. """ 

import json 
import argparse 

from typing import List 

import logging 
logger = logging.getLogger(__name__)



def load_tables_dict(tables_file: str, key: str = 'id'): 
    tables_dict = {} 
    with open(tables_file, 'r') as fr: 
        for line in fr: 
            tdict = json.loads(line.strip()) 
            table_id = tdict[key]
            tables_dict[table_id] = tdict 
    return tables_dict 


def get_annotated_table_ids(anno_path: str) -> List[str]: 
    gold_ids = [] 
    
    with open(anno_path, 'r') as fr: 
        for line in fr: 
            sample = json.loads(line.strip())
            gold_ids.append(sample['table']['tableId'])
            
    return gold_ids


def convert_ctxs(
    tables_dict, 
    retrieved_path: str, 
    converted_path: str, 
    gold_table_ids: List[str]
): 
    with open(retrieved_path, 'r') as fr: 
        dataset = json.load(fr) 
    
    newset = [] 
    for i, sample in enumerate(dataset): 
        pos_ctxs, neg_ctxs = [], []
        
        if gold_table_ids: 
            gold_tid = gold_table_ids[i] 
            gold_table = tables_dict[gold_tid] 
            
            table_ctx = {
                'id': gold_tid, 
                'title': gold_table['title'], 
                'score': 1.0, 
                'has_answer': True, 
                'table': gold_table, 
            }
            pos_ctxs.append(table_ctx)
        
        for j, ctx in enumerate(sample['ctxs']): 
            table_id = ctx['id']
            if ':' in table_id: 
                table_id = table_id[table_id.index(':')+1: ]
            ctab = tables_dict[table_id]
            assert ctab is not None
            
            table_ctx = {
                'id': ctx['id'], 
                'title': ctx['title'], 
                'score': ctx['score'], 
                'has_answer': ctx['has_answer'], 
                'table': ctab, 
            }
            
            if table_ctx['has_answer']: 
                pos_ctxs.append(table_ctx)
            else: 
                neg_ctxs.append(table_ctx) 
        
        num_hard_negatives = min(5, len(neg_ctxs))
        new_sample = {
            'question': sample['question'], 
            'answers': sample['answers'], 
            'positive_ctxs': pos_ctxs if pos_ctxs else neg_ctxs[:2], 
            'negative_ctxs': neg_ctxs[num_hard_negatives: ], 
            'hard_negative_ctxs': neg_ctxs[: num_hard_negatives], 
        }
        newset.append(new_sample)
    
    with open(converted_path, 'w') as fw: 
        json.dump(newset, fw) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    
    parser.add_argument('--tables_file', type=str, required=True) 
    parser.add_argument('--retrieved_path', type=str, required=True) 
    parser.add_argument('--converted_path', type=str, required=True) 
    parser.add_argument('--annotated_path', type=str, default=None) 
    
    args = parser.parse_args() 
    
    tables_dict = load_tables_dict(args.tables_file) 
    logger.info(f"loaded {len(tables_dict)} tables") 
    
    gold_table_ids = None
    if args.annotated_path: 
        gold_table_ids = get_annotated_table_ids(args.annotated_path)
 
    convert_ctxs(tables_dict, args.retrieved_path, args.converted_path, gold_table_ids)