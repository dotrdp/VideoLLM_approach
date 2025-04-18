from loader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, VisualBertModel


VisualBertVCR = VisualBertModel.from_pretrained("visualbertvcr")
print(VisualBertVCR.config)
class VisualBertEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = "A"