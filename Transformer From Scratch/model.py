import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        