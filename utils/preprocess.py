import math

import numpy as np
import pandas as pd
import torch
from pyproj import Proj
from scipy.spatial import distance_matrix

from utils.constants import *
from utils.dataset import TrajGPTDataset

#####################################################################
# File I/O
#####################################################################


def load_visit_df(city):
    data_path = f"/data/private/maria_data/data/trajfm_veraset_splits/veraset/Visits/{city}/whole_veraset_processed.parquet"
    df = pd.read_parquet(
        data_path,
        columns=[
            "user_id",
            "place_lon",
            "place_lat",
            "arrival_time",
            "departure_time",
            "place_id",
        ],
    )

    # Filter user
    # Each user should have at least 2 visits: one for input, the other for label
    groupby = df.groupby("user_id", as_index=True).size() >= 2
    users = groupby[groupby].index
    df = df[df["user_id"].isin(users)]

    # Sort and reset ("ignore") index
    df.sort_values(by=["user_id", "arrival_time"], inplace=True, ignore_index=True)
    df["id"] = df.index
    return df


#####################################################################
# UTM Conversion
#####################################################################
def get_utm_zone(longitude):
    return int((math.floor((longitude + 180) / 6) % 60) + 1)


def project_latlon_to_xy(df):
    # Get centroid
    lat_c, lon_c = df[["place_lat", "place_lon"]].mean().values

    # Initialize UTM projection
    zone_number = get_utm_zone(lon_c)
    project_utm = Proj(proj="utm", zone=zone_number, ellps="WGS84")

    # Project centroid to UTM
    utm_x_c, utm_y_c = project_utm(lon_c, lat_c)

    utm_x, utm_y = project_utm(df["place_lon"], df["place_lat"])
    x = utm_x - utm_x_c  # Unit: meter
    y = utm_y - utm_y_c  # Unit: meter
    return x, y


#####################################################################
# Data Processing
#####################################################################
def process_space_inplace(df):
    # Project (lat, lon) to utm in meters
    df["x"], df["y"] = project_latlon_to_xy(df)


def process_time_inplace(df):
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    df["departure_time"] = pd.to_datetime(df["departure_time"])

    # Convert time to relative time (unit: day)
    oldest_timestamp = df["arrival_time"].min()
    df["arrival_time"] = (df["arrival_time"] - oldest_timestamp).dt.total_seconds() / (
        60 * 60 * 24
    )
    df["departure_time"] = (
        df["departure_time"] - oldest_timestamp
    ).dt.total_seconds() / (60 * 60 * 24)

    # Calculate duration (unit: hour)
    df["duration"] = (df["departure_time"] - df["arrival_time"]) * 24

    # Calculate travel time (unit: hour)
    df["travel_time"] = (df["arrival_time"] - df["departure_time"].shift(1)) * 24
    # First row of each user does not have travel time
    df.loc[df.groupby("user_id").nth(0).index, "travel_time"] = 0


def split_indices_for_next_prediction(df, train_ratio, max_seq_len=20):
    df.sort_values(by=["user_id", "arrival_time"], inplace=True, ignore_index=True)
    train_first_indices = []
    val_first_indices = []
    train_last_indices = []
    val_last_indices = []
    for user_id, group in df.groupby("user_id"):
        num_instances = (len(group) + max_seq_len - 3) // max_seq_len
        for i in range(num_instances):
            start_idx = i * max_seq_len
            end_idx = min((i + 1) * max_seq_len, len(group))

            if i < int(num_instances * train_ratio):
                train_first_indices.append(group.index[start_idx])
                train_last_indices.append(group.index[end_idx - 1])
            else:
                val_first_indices.append(group.index[start_idx])
                val_last_indices.append(group.index[end_idx - 1])

    train_indices = pd.DataFrame(
        {"first_index": train_first_indices, "last_index": train_last_indices}
    )
    val_indices = pd.DataFrame(
        {"first_index": val_first_indices, "last_index": val_last_indices}
    )

    train_indices.reset_index(drop=True, inplace=True)
    val_indices.reset_index(drop=True, inplace=True)
    return train_indices, val_indices


def load_dataset(city, train_ratio):
    df = load_visit_df(city)
    process_space_inplace(df)
    process_time_inplace(df)

    train_indices, val_indices = split_indices_for_next_prediction(df, train_ratio)

    # Keep only necessary fields
    df = df[FIELDS]

    train_data = TrajGPTDataset(df, train_indices)
    val_data = TrajGPTDataset(df, val_indices)

    # Number of places
    # NOTE Some place are filtered out after filtering users.
    num_places = df["place_id"].max() + 1

    # Scale of place of interest (ROI) for Space2Vec
    lambda_max = calculate_pairwise_distance_upperbound(df)

    return (train_data, val_data, num_places, lambda_max)


def calculate_pairwise_distance_upperbound(df):
    """Compute scale of place of Interest (ROI)"""
    min_x, max_x = df["x"].min(), df["x"].max()
    min_y, max_y = df["y"].min(), df["y"].max()
    corners = np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
    dist_matrix = distance_matrix(corners, corners)
    return np.max(dist_matrix)


#####################################################################
# Visit Infilling
#####################################################################
def generate_dynamic_mask(batch, mask_probability=0.2):
    """Generate dynamic mask for visit infilling task.

    Args:
        batch (torch.FloatTensor): (N, L, D) tensor

    Returns:
        torch.BoolTensor: (N, L) tensor
    """
    N, L, D = batch.shape
    probability_matrix = torch.full((N, L), mask_probability)
    probability_matrix[:, 0] = 0.0
    probability_matrix[:, -1] = 0.0
    dynamic_mask = torch.bernoulli(probability_matrix).bool()
    return dynamic_mask


def get_offset_kvpair_tensorized(batch_size, tensor, mask, offsets=None):
    """Match indices between old and new tensors"""
    seq_len = tensor.shape[1]
    x = torch.arange(batch_size).unsqueeze(1).expand(batch_size, seq_len)
    y = torch.cumsum(mask, dim=1) - 1
    if offsets is not None:
        y += offsets.unsqueeze(1).expand(batch_size, seq_len)
    new_indices = torch.stack([x[mask], y[mask]], dim=-1).view(-1, 2)
    values = tensor[mask]
    return new_indices, values


def split_input_target(batch_dict):
    """Split batch dictionary into input and target dictionaries"""
    # NOTE: Don't shift target_dict because model.forward() does it
    # Keeping the last token in input because we do teacher forcing for joint prediction
    input_dict = {key: batch_dict[key] for key in IN_FIELDS}
    target_dict = {key: batch_dict[key][:, 1:] for key in OUT_FIELDS}
    return input_dict, target_dict


def convert_batch_to_model_io(batch, device):
    for k, v in batch.items():
        batch[k] = v.to(device)

    input_dict, target_dict = split_input_target(batch)
    return input_dict, target_dict


def compute_after_sep_mask(place_id):
    """Compute mask for the part after SEP token"""
    sep_mask = place_id == SEP
    after_sep_mask = torch.cumsum(sep_mask, dim=1) > 0
    return after_sep_mask
