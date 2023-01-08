import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from lightning_model import LightningPCModel

from pytorch_lightning.loggers import WandbLogger
import argparse
import transformers
import datetime
from pytorch_lightning.callbacks import LearningRateMonitor

import wandb
import warnings
import torch
from os.path import join as pjoin

warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

ROOT_DIR = os.getcwd()
MODEL_DIR = pjoin(ROOT_DIR, 'model_ckpt')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Reranking module based on PolyEncoder')
    parser.add_argument('--train',
                        action='store_true',
                        default=True,
                        help='for training')

    parser.add_argument('--data_dir',
                        type=str,
                        default='/opt/ml/input/data/train_dataset')

    parser.add_argument("--pretrained_model", type=str, default="klue/bert-base")
    parser.add_argument("--model_type", type=str, default="poly")
    parser.add_argument("--embed_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cand_size", type=int, default=31)
    parser.add_argument("--max_epoch", type=int, default=1)

    parser.add_argument("--pooling_method", type=str, default="first")

    parser.add_argument("--cuda", 
                        action='store_true',
                        default=True)

    parser.add_argument("--gpuid", nargs='+', type=str, default = 'cuda:0')

    today = datetime.datetime.now()

    parser.add_argument("--model_name", type=str, default=f"{today.strftime('%m%d')}_qa")
    parser.add_argument("--model_pt", type=str, default=f'{MODEL_DIR}/model_last.ckpt')

    parser = LightningPCModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = args.data_dir

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,use_fast=True)
    
    wandb.login()
    # project_name: testtest
    # entity: mrc_bora
    # exp_name: new baseline test
    wandb.init(project='10_poly_encoder', entity='mrc_bora', name='poly_encoder_real_final')
    print(args.pretrained_model)
    wandb_logger = WandbLogger(name='poly_encoder_loss_change', project='10_poly_encoder',log_model=True)
    #log_model =True : 마지막에 Log model checkpoints at the end of training
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.train:
        seed_everything(42)
        with torch.cuda.device(args.gpuid):
            checkpoint_callback = ModelCheckpoint(
                dirpath='model_ckpt',
                filename='{epoch:02d}-{train_loss:.2f}',
                verbose=True,
                save_last=True,
                save_top_k=3,
                monitor='train/loss',
                mode='min',
                prefix=f'{args.model_name}'
            )

            # model = LightningPCModel(args, tokenizer=tokenizer)
            model = LightningPCModel.load_from_checkpoint('model_ckpt/0103_qa-last.ckpt',hparams = args, tokenizer = tokenizer)
            model.train()
            trainer = Trainer(
                            callbacks =[lr_monitor,checkpoint_callback],
                            # check_val_every_n_epoch=0.2,                   #epoch 단위로 float 가능이기 때문에
                            log_every_n_steps = 10,
                            gpus = 1,
                            logger=wandb_logger,
                            max_epochs=args.max_epochs, 
                            num_processes=4,
                            amp_backend='native',
                            amp_level='O2',
                            num_sanity_val_steps=2
                            )
            
            trainer.fit(model)
            print('best model path {}'.format(checkpoint_callback.best_model_path))

    else:
        print('Evaluation')