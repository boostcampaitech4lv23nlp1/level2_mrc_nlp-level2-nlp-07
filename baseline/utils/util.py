import random
from typing import Callable, Dict, List, NoReturn, Tuple

import numpy as np
import torch
from arguments import (
    DataTrainingArguments, ModelArguments, inference_args_class, cfg,
    model_args, data_args, inference_args)
from model.sparse_retrieval import BM25SparseRetrieval,TFIDFSparseRetrieval
from model.dense_retrieval import DenseRetrieval
from model.poly_retrieval import polyRetrieval
from transformers import EvalPrediction,is_torch_available
from typing import List, NoReturn, Optional, Tuple, Union
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_metric,
    load_dataset
)
import pandas as pd
def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('lock_all_seed')
   

def compute_metrics(p: EvalPrediction) -> Dict:
    metric = load_metric("squad")
    return metric.compute(predictions=p.predictions, references=p.label_ids)


def run_sparse_retrieval(
    cfg,
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    data_path: str = "/opt/ml/input/data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    if cfg.test.retrieval == 'TF-IDF':
        retriever = TFIDFSparseRetrieval(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
        )
        retriever.get_sparse_embedding()
    elif cfg.test.retrieval == 'BM25':
        retriever = BM25SparseRetrieval(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
        )
        retriever.get_sparse_embedding()
    elif cfg.test.retrieval == 'DPR':
        retriever = DenseRetrieval(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
        )
        retriever.get_passage_embedding()
    elif cfg.test.retrieval == 'POLY':
        retriever = polyRetrieval(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
        )
        retriever.get_passage_embedding()
    else:
        assert f'Could not found retrieval {cfg.test.retrieval} or Not exist'
    

    if cfg.data.use_faiss:
        retriever.build_faiss(num_clusters=cfg.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=cfg.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if inference_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif inference_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets

def aug_data(data, aug_path):
    aug_dataset = data
    for i in aug_path:
        if i == 'squad_kor_v1':
            new_data = load_dataset("squad_kor_v1")
            train_df = Dataset.from_pandas(pd.DataFrame(new_data['train']))
            validation_df = Dataset.from_pandas(pd.DataFrame(new_data['validation']))
            train_data = concatenate_datasets([aug_dataset['train'],train_df])
            validation_data = concatenate_datasets([aug_dataset['validation'],validation_df])
            aug_dataset = DatasetDict({'train': train_data, 'validation' : validation_data})
        else:               # squad_kor_v1 제외하고 모두 경로 설정이기 때문에 path로 입력을 바꾸고 진행했습니다. 
            train = Dataset.from_json(f'{i}/train.json')
            valid = Dataset.from_json(f'{i}/valid.json')
            new_train = concatenate_datasets([aug_dataset['train'],train])
            new_valid = concatenate_datasets([aug_dataset['validation'],valid])
            aug_dataset = DatasetDict({'train' : new_train, 'validation' : new_valid})
    return aug_dataset