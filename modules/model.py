import torch.nn as nn

from modules.decoder import CausalMemoryDecoder
from modules.encoder import CausalEncoder
from modules.gmm import GMM
from modules.input import SourceInput


class TrajGPT(nn.Module):
    def __init__(
        self,
        num_places,
        sequence_len,
        lambda_max,
        num_heads=8,
        num_layers=4,
        num_gaussians=3,
        d_feedforward=32,
        d_embed=512,
        lambda_min=1e0,
    ):
        super().__init__()
        self.num_places = num_places
        self.d_model = (
            d_embed * 4
        )  # Four: location, arrival_time, departure_time, place_id

        # Sequence encoder
        self.input = SourceInput(num_places, d_embed, lambda_min, lambda_max)
        self.encoder = CausalEncoder(self.d_model, num_heads, num_layers, sequence_len)

        # place prediction
        self.place_id_decoder = CausalEncoder(self.d_model, num_heads, 1, sequence_len)
        self.place_id_head = nn.Linear(self.d_model, num_places)

        # Arrival (travel) prediction
        self.d_travel = d_embed * 2  # Two: place_id, location
        self.travel_decoder = CausalMemoryDecoder(
            self.d_travel, num_heads, sequence_len, d_feedforward
        )
        self.travel_head = GMM(self.d_travel, num_gaussians=num_gaussians)

        # Departure (duration) prediction
        self.d_duration = d_embed * 3  # Three: place_id, location, arrival_time
        self.duration_decoder = CausalMemoryDecoder(
            self.d_duration, num_heads, sequence_len, d_feedforward
        )
        self.duration_head = GMM(self.d_duration, num_gaussians=num_gaussians)

    def forward(self, kwargs):
        # Encode input sequence
        memory, tgt = self.encode_sequence(kwargs)
        # Predict place id
        place_id_out = self.predict_place(memory)
        # Predict travel time with teacher forcing
        travel_out = self.predict_travel_time(memory, tgt)
        # Predict duration with teacher forcing
        duration_out = self.predict_duration(memory, tgt)
        return {
            "place_id": place_id_out,
            "travel_time": travel_out,
            "duration": duration_out,
        }

    def encode_sequence(self, kwargs):
        seq = self.input(**kwargs)
        memory = self.encoder(
            seq[:, :-1, :]
        )  # (batch_size, src_seq_len == seq_len - 1, d_model)
        tgt = seq[:, 1:, :]  # (batch_size, tgt_seq_len == seq_len - 1, d_model)
        return memory, tgt

    def predict_place(self, memory):
        """memory: (batch_size, src_seq_len, d_model)"""
        place_id_dec = self.place_id_decoder(
            memory
        )  # (batch_size, src_seq_len, d_model)
        place_id_out = self.place_id_head(
            place_id_dec
        )  # (batch_size, src_seq_len, num_places)
        return place_id_out

    def predict_travel_time(self, memory, travel_tgt):
        """travel_tgt: (batch_size, tgt_seq_len, d_travel)"""
        travel_decoder_input = travel_tgt[..., : self.d_travel]
        travel_dec = self.travel_decoder(
            travel_decoder_input, memory[..., : self.d_travel]
        )  # (batch_size, tgt_seq_len, d_model)
        travel_out = self.travel_head(
            travel_dec
        )  # (batch_size, tgt_seq_len, num_gaussians) for key in ['weight', 'loc','scale']
        return travel_out

    def predict_duration(self, memory, duration_tgt):
        duration_decoder_input = duration_tgt[..., : self.d_duration]
        duration_dec = self.duration_decoder(
            duration_decoder_input, memory[..., : self.d_duration]
        )  # (batch_size, seq_len-1, d_model)
        duration_out = self.duration_head(
            duration_dec
        )  # (batch_size, seq_len-1, num_gaussians=3) for key in ['weight', 'loc','scale']
        return duration_out
