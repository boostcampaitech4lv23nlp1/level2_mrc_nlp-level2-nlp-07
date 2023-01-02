import os
import torch

from omegaconf import OmegaConf
import wandb
import argparse

from train import *
from inference import test
from utils.util import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"


if __name__ =='__main__':
    ## parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    ## set seed
    set_seed(cfg.train.seed)

    ## train
    if cfg.exp.train:
        torch.cuda.empty_cache()
        ## wandb login
        wandb.login()
        wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name)

        print('------------------- train start -------------------------')
        train(cfg)

        ## wandb finish
        wandb.finish()

    ## inference
    if cfg.exp.test:
        torch.cuda.empty_cache()
        print('--------------------- test start ----------------------')
        test(cfg)

    print('----------------- Finish! ---------------------')