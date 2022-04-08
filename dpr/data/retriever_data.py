import collections
import os 
import csv
import json
import logging
import pickle
from typing import Dict, List

import hydra
import jsonlines
import torch
from omegaconf import DictConfig

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    get_dpr_files,
    read_nq_tables_jsonl,
    split_tables_to_chunks,
)
from dpr.data.biencoder_data import (
    BiEncoderTable, 
    get_nq_table_files, 
    get_processed_table, 
    get_processed_table_wiki, 
    get_processed_table_wqt, 
) 

from dpr.utils.data_utils import normalize_question

logger = logging.getLogger(__name__)

TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"])


class QASample:
    def __init__(self, query: str, id, answers: List[str]):
        self.query = query
        self.id = id
        self.answers = answers


class RetrieverData(torch.utils.data.Dataset):
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined in download_data.py
        """
        self.file = file
        self.data_files = []

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        assert (
            len(self.data_files) == 1
        ), "RetrieverData source currently works with single files only. Files specified: {}".format(self.data_files)
        self.file = self.data_files[0]


class QASrc(RetrieverData):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file)
        self.data = None
        self.selector = hydra.utils.instantiate(selector) if selector else None
        self.special_query_token = special_query_token
        self.query_special_suffix = query_special_suffix

    def __getitem__(self, index) -> QASample:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _process_question(self, question: str):
        # as of now, always normalize query
        question = normalize_question(question)
        if self.query_special_suffix and not question.endswith(self.query_special_suffix):
            question += self.query_special_suffix
        return question


class CsvQASrc(QASrc):
    def __init__(
        self,
        file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
        data_range_start: int = -1,
        data_size: int = -1,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_col = question_col
        self.answers_col = answers_col
        self.id_col = id_col
        self.data_range_start = data_range_start
        self.data_size = data_size

    def load_data(self):
        super().load_data()
        data = []
        start = self.data_range_start
        # size = self.data_size
        samples_count = 0
        # TODO: optimize
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col])
                id = None
                if self.id_col >= 0:
                    id = row[self.id_col]
                samples_count += 1
                # if start !=-1 and samples_count<=start:
                #    continue
                data.append(QASample(self._process_question(question), id, answers))

        if start != -1:
            end = start + self.data_size if self.data_size != -1 else -1
            logger.info("Selecting dataset range [%s,%s]", start, end)
            self.data = data[start:end] if end != -1 else data[start:]
        else:
            self.data = data


class JsonlQASrc(QASrc):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_attr = question_attr
        self.answers_attr = answers_attr
        self.id_attr = id_attr

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                answers = jline[self.answers_attr] if self.answers_attr in jline else []
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class KiltCsvQASrc(CsvQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
        data_range_start: int = -1,
        data_size: int = -1,
    ):
        super().__init__(
            file,
            question_col,
            answers_col,
            id_col,
            selector,
            special_query_token,
            query_special_suffix,
            data_range_start,
            data_size,
        )
        self.kilt_gold_file = kilt_gold_file


class KiltJsonlQASrc(JsonlQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_attr: str = "input",
        answers_attr: str = "answer",
        id_attr: str = "id",
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            selector,
            question_attr,
            answers_attr,
            id_attr,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                out = jline["output"]
                answers = [o["answer"] for o in out if "answer" in o]
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class TTS_ASR_QASrc(QASrc):
    def __init__(self, file: str, trans_file: str):
        super().__init__(file)
        self.trans_file = trans_file

    def load_data(self):
        super().load_data()
        orig_data_dict = {}
        with open(self.file, "r") as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            id = 0
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                orig_data_dict[id] = (question, answers)
                id += 1
        data = []
        with open(self.trans_file, "r") as tfile:
            reader = csv.reader(tfile, delimiter="\t")
            for r in reader:
                row_str = r[0]
                idx = row_str.index("(None-")
                q_id = int(row_str[idx + len("(None-") : -1])
                orig_data = orig_data_dict[q_id]
                answers = orig_data[1]
                q = row_str[:idx].strip().lower()
                data.append(QASample(q, idx, answers))
        self.data = data


class CsvCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        logger.info("Reading file %s", self.file)
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                # for row in ifile:
                # row = row.strip().split("\t")
                if row[self.id_col] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col].strip('"')
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])


class KiltCsvCtxSrc(CsvCtxSrc):
    def __init__(
        self,
        file: str,
        mapping_file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file, id_col, text_col, title_col, id_prefix, normalize=normalize)
        self.mapping_file = mapping_file

    def convert_to_kilt(self, kilt_gold_file, dpr_output, kilt_out_file):
        logger.info("Converting to KILT format file: %s", dpr_output)

        with open(dpr_output, "rt") as fin:
            dpr_output = json.load(fin)

        with jsonlines.open(kilt_gold_file, "r") as reader:
            kilt_gold_file = list(reader)
        assert len(kilt_gold_file) == len(dpr_output)
        map_path = self.mapping_file
        with open(map_path, "rb") as fin:
            mapping = pickle.load(fin)

        with jsonlines.open(kilt_out_file, mode="w") as writer:
            for dpr_entry, kilt_gold_entry in zip(dpr_output, kilt_gold_file):
                # assert dpr_entry["question"] == kilt_gold_entry["input"]
                provenance = []
                for ctx in dpr_entry["ctxs"]:
                    wikipedia_id, end_paragraph_id = mapping[int(ctx["id"])]
                    provenance.append(
                        {
                            "wikipedia_id": wikipedia_id,
                            "end_paragraph_id": end_paragraph_id,
                        }
                    )
                kilt_entry = {
                    "id": kilt_gold_entry["id"],
                    "input": kilt_gold_entry["input"],  # dpr_entry["question"],
                    "output": [{"provenance": provenance}],
                }
                writer.write(kilt_entry)

        logger.info("Saved KILT formatted results to: %s", kilt_out_file)


class JsonlTablesCtxSrc(object):
    def __init__(
        self,
        file: str,
        tables_chunk_sz: int = 100,
        split_type: str = "type1",
        id_prefix: str = None,
    ):
        self.tables_chunk_sz = tables_chunk_sz
        self.split_type = split_type
        self.file = file
        self.id_prefix = id_prefix

    def load_data_to(self, ctxs: Dict):
        docs = {}
        logger.info("Parsing Tables data from: %s", self.file)
        tables_dict = read_nq_tables_jsonl(self.file)
        table_chunks = split_tables_to_chunks(tables_dict, self.tables_chunk_sz, split_type=self.split_type)
        for chunk in table_chunks:
            sample_id = self.id_prefix + str(chunk[0])
            docs[sample_id] = TableChunk(chunk[1], chunk[2], chunk[3])
        logger.info("Loaded %d tables chunks", len(docs))
        ctxs.update(docs)


# %% NQ-Table

# [table_sources]
class JsonlNQTablesCtxSrc(object):    # need a long time to process
    """To load NQ tables from one jsonl file. """

    def __init__(self, file: str, id_prefix: str = None,): 
        self.file = file 
        self.data_files = [] 

        self.id_prefix = id_prefix
    
    def load_data(self):
        self.data_files = get_nq_table_files(self.file)
        assert (len(self.data_files) == 1), \
            "JsonTablesCtxSrc works with single files only. Files specified: {}".format(self.data_files)
        self.file = self.data_files[0]
    
    def load_data_to(self, ctxs: Dict[object, BiEncoderTable], cfg: DictConfig):
        """Load dataset into the `ctxs` argument. """
        self.load_data() 
        logger.info("Loading NQ-Table data from: %s", self.file)
        with open(self.file, 'r') as fr: 
            dataset = [json.loads(line.strip()) for line in fr]

        for i, sample in enumerate(dataset): 
            if self.id_prefix: 
                sample_id = self.id_prefix + sample['tableId']
                # sample_id = self.id_prefix + str(i)
            else: 
                sample_id = sample['tableId']
                # sample_id = str(i)
            
            processed_table = get_processed_table(
                table=sample, 
                row_selection=cfg.row_selection, 
                max_cell_num=cfg.max_cell_num, 
                max_words=cfg.max_words, 
                max_words_per_header=cfg.max_words_per_header, 
                max_words_per_cell=cfg.max_words_per_cell, 
                max_cell_num_per_row=cfg.max_cell_num_per_row, 
                header_delimiter=cfg.header_delimiter, 
                cell_delimiter=cfg.cell_delimiter, 
                row_delimiter=cfg.row_delimiter, 
            )
            ctxs[sample_id] = processed_table

        logger.info("Loaded %d tables", len(ctxs))


# [table_sources]
class JsonlNqtCtxSrc(object):         # pre-processed version
    """class to load processed nq-tables from jsonl files. """
    def __init__(self, file: str, id_prefix: str = 'nqt:'): 
        self.file = file 
        self.id_prefix = id_prefix 
        
        self.data_files = [] 
    
    def load_data(self): 
        self.data_files = get_nq_table_files(self.file)
        assert (len(self.data_files) == 1), \
            "JsonTablesCtxSrc works with single files only. Files specified: {}".format(self.data_files)
        self.file = self.data_files[0]
    
    def load_data_to(self, ctxs: Dict[object, BiEncoderTable], cfg: DictConfig):
        """Load data from self.file to `ctxs`. """
        self.load_data() 
        logger.info("Loading NQ-Table data from: %s", self.file)
        
        with open(self.file, 'r') as fr: 
            dataset = [json.loads(line.strip()) for line in fr]
        # sample: {'id': str, 'title': str, 'cells': [{'text': str, 'row_idx': 0, 'col_idx': 0}, ..., {}]}
        for i, sample in enumerate(dataset): 
            if self.id_prefix: sample_id = self.id_prefix + sample['id']
            else: sample_id = sample['id']
            ctxs[sample_id] = BiEncoderTable(
                cells=sample['cells'], title=sample['title'], 
            )
        logger.info("Loaded %d tables", len(ctxs))
        
        
# [table_retrieval]
class JsonlQASrcTable(QASrc):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        question_attr: str = "originalText",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_attr = question_attr
        self.answers_attr = answers_attr
        self.id_attr = id_attr

    def _load_data(self):
        self.data_files = get_nq_table_files(self.file)
        assert (len(self.data_files) == 1), \
            "JsonTablesCtxSrc works with single files only. Files specified: {}".format(self.data_files)
        self.file = self.data_files[0]

    def load_data(self):
        self._load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                qa = jline['questions'][0]
                question = qa['originalText']
                answers = qa['answer']['answerTexts']
                id = qa['id']
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data
