"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import numpy as np
from datasets import (
    DatasetDict,
    load_from_disk,
    load_metric,
)

from trainer.trainer import QuestionAnsweringTrainer
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
)
from utils.load_data import MRC_Dataset
from utils.util import run_sparse_retrieval,set_seed
from model.reader import MRCModel
from utils.util import compute_metrics
import torch
logger = logging.getLogger(__name__)


def test(cfg):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", cfg)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(cfg.train.seed)

    datasets = load_from_disk(cfg.data.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_name
        if cfg.model.tokenizer_name is not None
        else cfg.model.model_name_or_path,
        use_fast=True,
    )
    if cfg.load_last_model:
        model = MRCModel(cfg.model_name_or_path)
        model.load_state_dict(torch.load(cfg.model.load_last_model))
        print(f"model is from {cfg.model.load_last_model}")


    # True일 경우 : run passage retrieval
    if cfg.exp.test:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize, datasets
        )

    eval_dataset = MRC_Dataset(datasets["validation"],tokenizer=tokenizer)
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if cfg.train.fp16 else None
    )

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=TrainingArguments,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=eval_dataset.post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if cfg.train.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )
        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )
