import torch
import torch.nn as nn

from modules.space2vec import Space2Vec
from modules.time2vec import Time2Vec
from utils.constants import N_SPECIAL_TOKENS, PAD


class SourceInput(nn.Module):
    def __init__(self, n_regions, d_embed, lambda_min, lambda_max):
        """
        Args:
            n_regions (int): The number of tokens in the vocabulary.
            sequence_length (int): The original maximum sequence length.
            d_embed (int): The dimension of the embeddings.
            lambda_min (float): The minimum scale for space2vec
            lambda_max (float): The maximum scale for space2vec
            pad_value (int): The padding value for the embeddings.
        """
        super().__init__()
        self.space2vec = Space2Vec(d_embed, lambda_min, lambda_max)
        self.time2vec = Time2Vec(d_embed)
        self.region_embedding = nn.Embedding(N_SPECIAL_TOKENS + n_regions, d_embed, padding_idx=PAD)

    def forward(self, region_id, x, y, arrival_time, departure_time):
        """
        Each input variable is a tensor of size (batch_size, seq_len)
        """
        locations = torch.stack([x, y], dim=-1)
        loccation_encoding = self.space2vec(locations)
        arrival_encoding = self.time2vec(arrival_time)
        departure_encoding = self.time2vec(departure_time)
        region_embedding = self.region_embedding(region_id)
        visit_embedding = torch.concat([
            loccation_encoding,
            arrival_encoding,
            departure_encoding,
            region_embedding
        ], dim=-1)  # (batch_size, seq_len, d_embed*4)
        return visit_embedding
