from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn,Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math,os,torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):
  def __init__(self,ntoken:int,d_model:int,nhead:int,d_hid:int,nlayers:int, dropout:float = 0.5):
    super().__init__()
    self.model_type = 'Transformer'
    self.pos_encoder = PositionalEncodeing(d_model, dropout)

    self.encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, nlayers)
    self.embedding = nn.Embedding(ntoken, d_model)
    self.d_model = d.model
    self.linear = nn.Linear(d_model, ntoken)
    self.init_weights()


  def init_weights(self) -> None:
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange,initrange)
    self.linear.bias.data.zero_()
    self.linear.weight.data.uniform_(-initrange,initrange)

  def forward(self, src:Tensor, src_mask: Tensor = None) -> Tensor:
    src = self.embedding(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src)
    output = self.transformer_encoder(src,src_mask)
    output = self.linear(output)
    return output



class PositionalEncoding(nn.Module)
  def __init__(self,d_model:int, dropout:float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/ d_model))

    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)

    self.register_buffer('pe',pe)
  def forward(self, x:Tensor) -> Tensor:

    x = x + self.pe[:,x.size(0)]

    return self.dropout(x)

  
