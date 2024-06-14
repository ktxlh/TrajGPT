# Special tokens
N_SPECIAL_TOKENS = 4
PAD, BLANK, SEP, ANS = list(range(N_SPECIAL_TOKENS))

# Tasks
NEXT_PREDICTION = 0
INFILLING = 1

# Sequence length
RAW_SEQ_LEN = 128
SEQ_LEN = {
    INFILLING: RAW_SEQ_LEN * 3,
    NEXT_PREDICTION: RAW_SEQ_LEN,
}

# Metrics
TOP_KS = [1, 5, 10, 20]
P_WITHIN_T = [5, 10, 20]

# Data
IN_FIELDS = ['x', 'y', 'region_id', 'arrival_time', 'departure_time']
OUT_FIELDS = ['region_id', 'travel_time', 'duration']
FIELDS = ['x', 'y', 'region_id', 'arrival_time', 'departure_time', 'duration', 'travel_time']

MAX_VALID_TRAVEL_TIME = 4  # hour
