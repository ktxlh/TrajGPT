import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from modules.positional_encoding import PositionalEncoding


class CausalEncoder(nn.Module):
    """Causal Transformer Encoder"""
    def __init__(self, d_model, num_heads, num_layers, sequence_len):
        """
        Args:
            d_embed (int): The dimension of the embeddings.
            num_heads (int): The number of heads in the multi-head attention mechanism.
            num_layers (int): The number of layers in the transformer encoder.
            sequence_len (int): The maximum sequence length.
        """
        super().__init__()

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers,
        )
        self.pos_encoder = PositionalEncoding(d_model)
        src_mask = nn.Transformer.generate_square_subsequent_mask(sequence_len)
        self.register_buffer('src_mask', src_mask, persistent=False)

    def forward(self, src):
        """
        Input is a tensor of size (batch_size, seq_len, d_model)
        """
        # Generate a causal modeling mask
        src_mask = self.src_mask[:src.shape[1], :src.shape[1]]
        output = self.transformer_encoder(self.pos_encoder(src), mask=src_mask, is_causal=True)
        return output  # (batch_size, seq_len, d_model)