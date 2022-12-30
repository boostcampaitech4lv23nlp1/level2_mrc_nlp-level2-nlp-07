from dataclasses import dataclass, field
from typing import Optional
from omegaconf import OmegaConf
import argparse
from transformers import (
    HfArgumentParser,
    TrainingArguments
)

## parser
parsers = argparse.ArgumentParser()
parsers.add_argument('--config', type=str, default='config')
args, _ = parsers.parse_known_args()
cfg = OmegaConf.load(f'./config/{args.config}.yaml')

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        default=cfg.model.model_name,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    trained_model_name: str = field(
        default=cfg.data.output_model_dir,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=cfg.data.dataset_name,
        metadata={"help": "The name of the dataset to use."},
    )

    test_dataset_name: Optional[str] = field(
        default=cfg.data.test_dataset_name,
        metadata={"help": "The name of the dataset to use."},
    )
    
    overwrite_cache: bool = field(
        default=cfg.data.overwrite_cache,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=cfg.data.max_seq_length,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=cfg.data.pad_to_max_length,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=cfg.data.doc_stride,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=cfg.data.max_answer_length,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=cfg.data.eval_retrieval,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    dense_retrieval: bool = field(
        default=cfg.data.dense_retrieval,
        metadata={"help": "Whether to run passage retrieval using dense embedding."},
    )
    num_clusters: int = field(
        default=cfg.data.num_clusters, 
        metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=cfg.data.top_k_retrieval,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=cfg.data.use_faiss, 
        metadata={"help": "Whether to build with faiss"}
    )

@dataclass
class training_args_class:
    args: str = field( 
            default = TrainingArguments(
                do_train = True,
                do_eval = True,
                do_predict = False,
                evaluation_strategy = "steps",
                eval_steps = cfg.train.eval_step,
                fp16 = True,
                gradient_accumulation_steps = cfg.train.gradient_accumulation_steps,
                label_smoothing_factor = cfg.train.label_smoothing_factor,
                learning_rate = cfg.train.lr,
                logging_strategy = "steps",
                logging_steps = cfg.train.logging_step,
                load_best_model_at_end = cfg.train.load_best_model_at_end,
                metric_for_best_model = "eval_exact_match",
                num_train_epochs = cfg.train.epoch,
                output_dir = cfg.data.output_model_dir,
                per_device_train_batch_size = cfg.train.batch_size,
                per_device_eval_batch_size = cfg.train.batch_size,
                save_strategy = "steps",
                save_steps = 100,
                save_total_limit = 3,
                seed = cfg.train.seed,
                warmup_ratio = cfg.train.warmup_ratio,
                weight_decay = cfg.train.weight_decay,
            )
    )
@dataclass
class inference_args_class:
    arg: str = field( 
            default = TrainingArguments(
                do_train = False,
                do_eval = False,
                do_predict = True,
                fp16 = True,
                gradient_accumulation_steps = cfg.train.gradient_accumulation_steps,
                label_smoothing_factor = cfg.train.label_smoothing_factor,
                learning_rate = cfg.train.lr,
                logging_strategy = "steps",
                logging_steps = cfg.train.logging_step,
                metric_for_best_model = "eval_exact_match",
                num_train_epochs = cfg.train.epoch,
                output_dir = cfg.data.output_json_dir,
                per_device_train_batch_size = cfg.train.batch_size,
                per_device_eval_batch_size = cfg.train.batch_size,
                save_strategy = "steps",
                save_steps = 100,
                save_total_limit = 3,
                seed = cfg.train.seed,
                warmup_ratio = cfg.train.warmup_ratio,
                weight_decay = cfg.train.weight_decay,
            )
    )
    
## parser
parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, training_args_class, inference_args_class)
    )
model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()