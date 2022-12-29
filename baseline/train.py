import logging
import sys
from typing import NoReturn

import os
from datasets import DatasetDict, load_from_disk, load_metric
from trainer.trainer import QuestionAnsweringTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    TrainingArguments,
)
from utils.load_data import MRC_Dataset
from utils.util import compute_metrics
from model.reader import MRCModel
from torch.optim.lr_scheduler import _LRScheduler,CosineAnnealingWarmRestarts
from torch import optim
import torch

logger = logging.getLogger(__name__)


def train(cfg):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)

    ##################### MODEL Define ########################

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
        use_fast=True,
    )
    if cfg.model.load_last_model:
        model = MRCModel(cfg.model.model_name_or_path)
        model.load_state_dict(torch.load(cfg.model.checkpoint_path))
        print(f"model is from {cfg.model.checkpoint_path}")

    else:
        # model = MRCModel(cfg.model.model_name_or_path)
        model = AutoModelForQuestionAnswering.from_pretrained(cfg.model.model_name_or_path)
        print(f"model is from {cfg.model.model_name_or_path}")

    ##################### DATA Define ########################
    datasets = load_from_disk(cfg.data.train_path)
    print(datasets)
    train_dataset = MRC_Dataset(datasets['train'],tokenizer=tokenizer,cfg=cfg)
    validation_dataset = MRC_Dataset(datasets['validation'],tokenizer=tokenizer,cfg=cfg)
    print(f'train_dataset {train_dataset[0].keys()}')
    print(f'validation_dataset {validation_dataset[0].keys()}')

    ##################### Trainer Define ########################
    # optimizer = optim.AdamW([
    #             {'params': model.plm.parameters()},
    #             {'params': model.dense_for_cls.parameters(), 'lr': cfg.train.second_lr},
    #             {'params': model.dense_for_e1.parameters(), 'lr': cfg.train.second_lr},
    #             {'params': model.dense_for_e2.parameters(), 'lr': cfg.train.second_lr},
    #             {'params': model.entity_classifier.parameters(), 'lr': cfg.train.second_lr}
    #                 ], lr=cfg.train.lr,weight_decay=0.01,eps = 1e-8)
    optimizer = optim.AdamW(model.parameters(),lr = cfg.train.lr, weight_decay=cfg.train.weight_decay,eps =1e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler.T_0, T_mult=cfg.scheduler.T_mult, eta_min=cfg.scheduler.eta_min)
    optimizers = (optimizer,scheduler)

    training_args = TrainingArguments(
        do_train = cfg.exp.train,
        do_eval = cfg.exp.train,
        do_predict = cfg.exp.test,
        output_dir=cfg.model.save_path,
        save_total_limit=5,
        save_steps=20, 
        num_train_epochs=cfg.train.epoch,
        learning_rate= cfg.train.lr,                         # default : 5e-5
        
        label_smoothing_factor = 0.1,
        gradient_accumulation_steps = cfg.train.gradient_accumulation_steps, 
        per_device_train_batch_size=cfg.train.batch_size,    # default : 16
        per_device_eval_batch_size=cfg.train.batch_size,     # default : 16

        warmup_steps=cfg.train.warmup_step,               
        weight_decay=cfg.train.weight_decay,
        warmup_ratio = cfg.train.warmup_ratio,           
    
        # for log
        logging_steps=cfg.train.logging_step,               
        evaluation_strategy='steps',     
        eval_steps = cfg.train.eval_step,                 # evaluation step.
        load_best_model_at_end = True,
        
        metric_for_best_model= 'eval_loss',
        greater_is_better=False,                             # False : loss 기준으로 최적화 해봄 도르
        # dataloader_num_workers=cfg.data.num_worker,
        fp16=cfg.train.fp16,


        # wandb
        report_to="wandb",
        run_name= cfg.wandb.exp_name
        )
    logger.info("Training/evaluation parameters %s", training_args)
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )


    trainer = QuestionAnsweringTrainer(
        model=model,                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,              # training arguments, defined above
        train_dataset= train_dataset,  # training dataset
        eval_dataset= validation_dataset,     # evaluation dataset use dev
        eval_examples = datasets['validation'],     # evaluation dataset use dev
        data_collator = data_collator,
        compute_metrics=compute_metrics,  # define metrics function
        post_process_function=validation_dataset.post_processing_function,
        optimizers = optimizers
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.train.patience)]# total_step / eval_step : max_patience
    )

    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(validation_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ## train model
    # trainer.train()
    
    ## save model
    # model.save_model(cfg.model.saved_model)
    # torch.save(model.state_dict(), PATH)
    torch.save(model.state_dict(),cfg.model.save_path)