import torch.nn as nn

from modules.decoder import CausalMemoryDecoder
from modules.encoder import CausalEncoder
from modules.gmm import GMM
from modules.input import SourceInput


class TrajGPT(nn.Module):
    def __init__(self, n_regions, sequence_length, lambda_max, 
                 n_heads=2, n_layers=4, n_gaussians=3, d_feedforward=32, d_embed=32, lambda_min=1e0):
        super().__init__()
        self.n_regions = n_regions
        self.d_model = d_embed * 4  # Four: location, arrival_time, departure_time, region_id

        # Sequence encoder
        self.input = SourceInput(n_regions, d_embed, lambda_min, lambda_max)
        self.encoder = CausalEncoder(self.d_model, n_heads, n_layers, sequence_length)

        # Region prediction
        self.region_id_decoder = CausalEncoder(self.d_model, n_heads, 1, sequence_length)
        self.region_id_head = nn.Linear(self.d_model, n_regions)

        # Arrival (travel) prediction
        self.d_travel = d_embed * 2  # Two: region_id, location
        self.travel_decoder = CausalMemoryDecoder(self.d_travel, n_heads, sequence_length, d_feedforward)
        self.travel_head = GMM(self.d_travel, ncomp=n_gaussians)

        # Departure (duration) prediction
        self.d_duration = d_embed * 3  # Three: region_id, location, arrival_time
        self.duration_decoder = CausalMemoryDecoder(self.d_duration, n_heads, sequence_length, d_feedforward)
        self.duration_head = GMM(self.d_duration, ncomp=n_gaussians)
        
    def forward(self, kwargs):
        # Encode input sequence
        memory, tgt = self.encode_sequence(kwargs)
        # Predict region id
        region_id_out = self.predict_region(memory)
        # Predict travel time with teacher forcing
        travel_out = self.predict_travel_time(memory, tgt)
        # Predict duration with teacher forcing
        duration_out = self.predict_duration(memory, tgt)
        return {
            'region_id': region_id_out, 
            'travel_time': travel_out, 
            'duration': duration_out
        }

    def encode_sequence(self, kwargs):
        seq = self.input(**kwargs)
        memory = self.encoder(seq[:, :-1, :])  # (batch_size, src_seq_len == seq_len - 1, d_model)
        tgt = seq[:, 1:, :]  # (batch_size, tgt_seq_len == seq_len - 1, d_model)
        return memory, tgt
            
    def predict_region(self, memory):
        """memory: (batch_size, src_seq_len, d_model)"""
        region_id_dec = self.region_id_decoder(memory)  # (batch_size, src_seq_len, d_model)
        region_id_out = self.region_id_head(region_id_dec)  # (batch_size, src_seq_len, n_regions)
        return region_id_out
    
    def predict_travel_time(self, memory, travel_tgt):
        """travel_tgt: (batch_size, tgt_seq_len, d_travel)"""
        travel_decoder_input = travel_tgt[..., :self.d_travel]
        travel_dec = self.travel_decoder(travel_decoder_input, memory[..., :self.d_travel])  # (batch_size, tgt_seq_len, d_model)
        travel_out = self.travel_head(travel_dec)  # (batch_size, tgt_seq_len, ncomp=1) for key in ['weight', 'loc','scale']        
        return travel_out
    
    def predict_duration(self, memory, duration_tgt):
        duration_decoder_input = duration_tgt[..., :self.d_duration]
        duration_dec = self.duration_decoder(duration_decoder_input, memory[..., :self.d_duration])  # (batch_size, seq_len-1, d_model)
        duration_out = self.duration_head(duration_dec)  # (batch_size, seq_len-1, ncomp=3) for key in ['weight', 'loc','scale']        
        return duration_out
