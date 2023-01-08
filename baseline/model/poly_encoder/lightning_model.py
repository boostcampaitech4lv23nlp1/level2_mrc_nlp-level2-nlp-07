import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch import optim

from .dataloader import IRData
from .model.poly_enc import PolyEncoder
from .model.bi_enc import BiEncoder
from .model.cross_enc import CrossEncoder
from transformers import DataCollatorWithPadding
from datasets import load_metric

class LightningPCModel(LightningModule):
    def __init__(self, tokenizer, **kwargs):
        super(LightningPCModel, self).__init__()
        self.hparams = self.add_model_specific_args()
        self.tok = tokenizer

        if self.hparams.model_type == 'poly':
            self.pc_model = PolyEncoder(self.hparams, model_name=self.hparams.pretrained_model, pooling_method=self.hparams.pooling_method)
        elif self.hparams.model_type == 'bi':
            self.pc_model = BiEncoder(self.hparams, model_name=self.hparams.pretrained_model, pooling_method=self.hparams.pooling_method)
        elif self.hparams.model_type == 'cross':
            self.pc_model = CrossEncoder(self.hparams, model_name=self.hparams.pretrained_model, pooling_method=self.hparams.pooling_method)
        
        self.criterion = self.dpr_loss
    def dpr_loss(self,scores):
        eps = 1e-9
        loss = -(torch.exp(scores[0])/(torch.exp(scores).sum()+ eps)).log() + eps
        return loss

    @staticmethod
    def add_model_specific_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_len',
                            type=int,
                            default=512)
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.01,
                            help='warmup ratio')
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
        args, _ = parser.parse_known_args()
        return args

    def forward(self, query, cand):
        score = self.pc_model(query, cand)
        return score

    def training_step(self, batch, batch_idx):
        query, cands, label = batch    
        query = query.unsqueeze(0)
        cands = cands.unsqueeze(0)
        label = label.unsqueeze(0) 
        # output must be 
        # query : batch, max_sq_len
        # cands : batch, cand_size, max_sq_len
        # label : batch, cand_size
        scores = []
        for i in range(self.hparams.cand_size + 1):
            cand_i = cands[:, i, :] # (batch_size, max_sq_len)
            score = self(query, cand_i)

            scores.append(score.squeeze(1))

        scores = torch.stack(scores, dim=0)
        # train_loss = scores.mean()
        # train_loss = self.criterion(scores, label)
        train_loss = self.criterion(scores)

        self.log('train/loss', train_loss, prog_bar=True, on_step=True, on_epoch=True)
        return train_loss


    def validation_step(self, batch, batch_idx):
        query, cands, label = batch    
        query = query.unsqueeze(0)
        cands = cands.unsqueeze(0)
        label = label.unsqueeze(0)
        scores = []
        for i in range(self.hparams.cand_size + 1):
            cand_i = cands[:, i, :]
            score = self(query, cand_i)
            # print(score.log())
            scores.append(score.squeeze(1))
        # scores = [self.criterion(score, label[:,i].unsqueeze(1)) for i,score in enumerate(scores)]
        scores = torch.stack(scores, dim=0)

        val_loss = self.criterion(scores.squeeze(0))

        # val_recall = self.recall(scores.squeeze(), label.squeeze())
        self.log('val/loss',val_loss ,prog_bar=True, on_step=True, on_epoch=True)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_losses = []
        for loss_avg in outputs:
            avg_losses.append(loss_avg)
        self.log('val/end_loss', torch.stack(avg_losses).mean(), prog_bar=True, on_epoch=True)
    
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr)
        # Cosine Annealing Learning Rate
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=1000,
            eta_min=1e-8
        )
        lr_scheduler =  {'scheduler': scheduler, 'name': 'CyclicLR',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        # output must be 
        # query : batch, max_sq_len
        # cands : batch, cand_size, max_sq_len
        # label : batch, cand_size
        data_collator = DataCollatorWithPadding(self.tok,padding=True, pad_to_multiple_of=8)
        query, cands, label = batch[0]
        cand = []
        for i in cands: # 11 times
            cand.append(i)
        cands = data_collator({'input_ids': cands})['input_ids']

        return torch.tensor(query),cands,torch.FloatTensor(label)

    def train_dataloader(self):
        self.train_set = IRData(data_path=f"{self.hparams.data_dir}/train", tokenizer=self.tok,cand_size=self.hparams.cand_size, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=1,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def val_dataloader(self):
        self.valid_set = IRData(data_path=f"{self.hparams.data_dir}/validation", tokenizer=self.tok,cand_size=self.hparams.cand_size, max_len=self.hparams.max_len)
        val_dataloader = DataLoader(
            self.valid_set, batch_size=1,
            shuffle=True, collate_fn=self._collate_fn)
        return val_dataloader