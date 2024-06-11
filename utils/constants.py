# Special tokens
PAD = 0
BLANK = 1
SEP = 2
ANS = 3
N_SPECIAL_TOKENS = 4

# Tasks
NEXT_PREDICTION = 0
INFILLING = 1

# Sequence length
SEQ_LEN = {
    NEXT_PREDICTION: 128,
    INFILLING: 128*3,
}

# Metrics
TOP_KS = [1, 5, 10, 20]
P_WITHIN_T = [5, 10, 20]

# Data
IN_FIELDS = ['x', 'y', 'region_id', 'arrival_time', 'departure_time']
OUT_FIELDS = ['region_id', 'travel_time', 'duration']
FIELDS = ['x', 'y', 'region_id', 'arrival_time', 'departure_time', 'duration', 'travel_time']
