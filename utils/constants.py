# Sequence length
RAW_SEQ_LEN = 128

# Data
IN_FIELDS = ["x", "y", "place_id", "arrival_time", "departure_time"]
OUT_FIELDS = ["place_id", "travel_time", "duration", "padding_mask"]
FIELDS = [
    "x",
    "y",
    "place_id",
    "arrival_time",
    "departure_time",
    "duration",
    "travel_time",
]
