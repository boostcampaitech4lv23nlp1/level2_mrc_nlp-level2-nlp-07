import os
import torch

from omegaconf import OmegaConf
import wandb
import argparse

from train import *
from inference import test
from utils.util import set_seed
import os
import torch
from inference import *
import wandb
from arguments import cfg, training_args
from train import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"


def main():
    torch.cuda.empty_cache()

    ## set sedd
    set_seed(training_args.train_args.seed)

    ## train
    if cfg.train.train_mode:
            ## wandb login
            wandb.login()
            wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name)
            
            print('---------------------- train start -------------------------')
            train()

            ## wandb finish
            wandb.finish()

    ## inference
    if cfg.test.test_mode:
        print('--------------------- test start ----------------------')
        test()
        
    print('----------------- Finish! ---------------------')
if __name__ == '__main__':
    main()