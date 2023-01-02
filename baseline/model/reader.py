
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering
    )
from torch import nn
import torch
from collections import OrderedDict
import numpy as np

class MRCModel(nn.Module):
    def __init__(
        self,
        pretrained_id: str = "klue/roberta-large",
        device = 'cuda:0'
    ):
        super(MRCModel, self).__init__()
        if pretrained_id:
            model_config= AutoConfig.from_pretrained(pretrained_id)
            self.plm = AutoModelForQuestionAnswering.from_pretrained(pretrained_id)
        
        self.hidden_size = model_config.hidden_size
        self.device = device
    def forward(self,inputs):
        output = self.plm(**inputs)
        return output