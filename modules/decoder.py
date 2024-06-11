import torch.nn.functional as F
from torch.nn import (Dropout, LayerNorm, Linear, Module, MultiheadAttention,
                      Transformer, TransformerDecoderLayer)


class CausalMemoryDecoder(TransformerDecoderLayer):
    """Memory-causal Transformer decoder layer without self-attention"""
    def __init__(self, d_model: int, n_heads: int, sequence_length: int, 
                 d_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, bias: bool = True) -> None:
        Module.__init__(self)

        self.multihead_attn = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True, bias=bias)
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, d_feedforward, bias=bias)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_feedforward, d_model, bias=bias)

        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        self.activation = F.relu

        causal_memory_mask = Transformer.generate_square_subsequent_mask(sequence_length)
        self.register_buffer("causal_memory_mask", causal_memory_mask, persistent=False)


    def forward(self, tgt, memory):
        """
        - memory: (batch_size, src_seq_len, d_model)
        - tgt: (batch_size, tgt_seq_len, d_model)
        - memory_mask: (tgt_seq_len, src_seq_len)
        """
        memory_mask = self.causal_memory_mask[:memory.shape[1], :memory.shape[1]]

        x = tgt
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, key_padding_mask=None))
        x = self.norm3(x + self._ff_block(x))
        return x
