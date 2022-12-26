import pickle as pickle
import os
import torch
import numpy as np
from inference import *
import wandb
from arguments import cfg, training_args
from train import *

if __name__ == "__main__":

    ## Reset the Memory
    torch.cuda.empty_cache()

    ## set sedd
    set_seed(training_args.args.seed)

    ## train
    if cfg.train.train_mode:
        if cfg.wandb.wandb_mode:
            ## wandb login
            wandb.login()
            wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name)
            
            print('---------------------- train start -------------------------')
            train()

            ## wandb finish
            wandb.finish()
        else:
            train()

    ## inference
    if cfg.test.test_mode:
        print('--------------------- test start ----------------------')
        test()
        
    print('----------------- Finish! ---------------------')