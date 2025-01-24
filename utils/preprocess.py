import math
from datetime import datetime

import h3
import numpy as np
import pandas as pd
import pytz
import torch
import trackintel
from pyproj import Proj
from scipy.spatial import distance_matrix

from utils.constants import *
from utils.dataset import TrajGPTDataset


#####################################################################
# File I/O
#####################################################################
def generate_geolife_visits():
    # Load the Geolife dataset
    geolife_path = "data/Geolife Trajectories 1.3/Data"
    pfs, _ = trackintel.io.read_geolife(geolife_path, print_progress=True)
    _, sp = pfs.generate_staypoints(print_progress=True)
    # Save visits
    trackintel.io.write_staypoints_csv(sp, 'data/geolife_staypoints.csv')


def load_geolife_visit_df():
    df = pd.read_csv('data/geolife_staypoints.csv')

    # Filter date
    df['arrival_time'] = pd.to_datetime(df['started_at'])
    df['departure_time'] = pd.to_datetime(df['finished_at'])
    start_mask = df['arrival_time'] >= datetime(2007, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
    finish_mask = df['departure_time'] <= datetime(2008, 12, 31, 23, 59, 59, tzinfo=pytz.UTC)
    df = df[start_mask & finish_mask]

    # Filter user
    # Each user should have at least 2 visits: one for input, the other for label
    groupby = (df.groupby('user_id', as_index=True).size() >= 20)
    users = groupby[groupby].index
    df = df[df['user_id'].isin(users)]

    # Sort and reset ("ignore") index
    df.sort_values(by=['user_id', 'started_at'], inplace=True, ignore_index=True)
    df['id'] = df.index
    return df


#####################################################################
# UTM Conversion
#####################################################################
def get_utm_zone(longitude):
    return int((math.floor((longitude + 180) / 6) % 60) + 1)


def project_latlon_to_xy(df):
    # Get centroid
    lat_c, lon_c = df[['latitude', 'longitude']].mean().values

    # Initialize UTM projection
    zone_number = get_utm_zone(lon_c)
    project_utm = Proj(proj='utm', zone=zone_number, ellps='WGS84')

    # Project centroid to UTM
    utm_x_c, utm_y_c = project_utm(lon_c, lat_c)

    utm_x, utm_y = project_utm(df['longitude'], df['latitude'])
    x = utm_x - utm_x_c  # Unit: meter
    y = utm_y - utm_y_c  # Unit: meter
    return x, y


#####################################################################
# Data Processing
#####################################################################
def process_space_inplace(df):
    # Convert str "POINT (longitude latitude)" to float
    pattern = r'POINT|\(|\)'
    df['geom'] = df['geom'].str.replace(pattern, "", regex=True)
    coordinates = df['geom'].str.split(expand=True).set_axis(['longitude', 'latitude'], axis=1)
    df['latitude'] = coordinates['latitude'].astype(float)
    df['longitude'] = coordinates['longitude'].astype(float)

    # Discretize locations
    df['h3_index'] = df.apply(lambda x: h3.geo_to_h3(x['latitude'], x['longitude'], resolution=7), axis=1)
    # The first N_SPECIAL_TOKENS indices are reserved for special tokens
    df['region_id'] = df['h3_index'].astype("category").cat.codes + N_SPECIAL_TOKENS

    # Project (lat, lon) to utm in meters
    df['x'], df['y'] = project_latlon_to_xy(df)


def process_time_inplace(df):
    # Convert time to relative time (unit: day)
    oldest_timestamp = df['arrival_time'].min()
    df['arrival_time'] = (df['arrival_time'] - oldest_timestamp).dt.total_seconds() / (60*60*24)
    df['departure_time'] = (df['departure_time'] - oldest_timestamp).dt.total_seconds() / (60*60*24)

    # Calculate duration (unit: hour)
    df['duration'] = (df['departure_time'] - df['arrival_time']) * 24

    # Calculate travel time (unit: hour)
    df['travel_time'] = (df['arrival_time'] - df['departure_time'].shift(1))
    # First row of each user does not have travel time
    df.loc[df.groupby('user_id').nth(0).index, 'travel_time'] = 0

    # Clip duration outliers
    max_duration = df['duration'].quantile(0.99)
    df.loc[df['duration'] > max_duration, 'duration'] = max_duration

    # Clip travel time outliers
    max_travel_time = df['travel_time'].quantile(0.99)
    df.loc[df['travel_time'] > max_travel_time, 'travel_time'] = max_travel_time

    return max_duration, max_travel_time


def split_indices_for_next_prediction(df):
    # Identify instances for next visit prediction
    groupby_user = df.groupby('user_id')

    # Find the first index of each instance
    # 1. First instance of each user
    user_first_indices = groupby_user.nth(0).index
    # 2. Instance with at least RAW_SEQ_LEN visits
    max_num_visits_per_user = groupby_user.size().max()
    long_enough_indices = groupby_user.nth(list(range(-max_num_visits_per_user, -RAW_SEQ_LEN))).index
    # 3. Combine the two
    first_indices = user_first_indices.union(long_enough_indices)

    # Find the corresponding last indices
    # 1. Last index of each user
    each_user_last_index = groupby_user.nth(-1).set_index('user_id')['id']
    # 2. Match it with first_indices
    user_last_indices = df.loc[first_indices]['user_id'].apply(lambda user_id: each_user_last_index.loc[user_id]).values
    # 3. Last index of each rolling window
    rolling_window_last_indices = (first_indices + RAW_SEQ_LEN - 1).values
    last_indices = np.minimum(rolling_window_last_indices, user_last_indices)

    # Sort instances by arrival_time
    index_df = pd.DataFrame({'first_index': first_indices, 'last_index': last_indices, 'arrival_time': df.loc[first_indices]['arrival_time']})
    index_df.sort_values(by='arrival_time', inplace=True, ignore_index=True)
    index_df.drop(columns=['arrival_time'], inplace=True)

    # Split instances into train, validation, and test sets
    train_size = int(0.8 * len(index_df))
    val_size = int(0.1 * len(index_df))
    train_indices = index_df[:train_size].reset_index(drop=True)
    val_indices = index_df[train_size: train_size + val_size].reset_index(drop=True)
    test_indices = index_df[train_size + val_size:].reset_index(drop=True)

    return train_indices, val_indices, test_indices


def split_index_df_for_infilling(index_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1
    
    # Shuffle the DataFrame to ensure randomness
    index_df = index_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the number of user_ids in each split
    user_ids = index_df['user_id'].unique()
    total_users = len(user_ids)
    train_size = int(total_users * train_ratio)
    val_size = int(total_users * val_ratio)
    
    # Split the user_ids
    train_user_ids = user_ids[:train_size]
    val_user_ids = user_ids[train_size:train_size + val_size]
    test_user_ids = user_ids[train_size + val_size:]
    
    # Create the split DataFrames
    train_indices = index_df[index_df['user_id'].isin(train_user_ids)].reset_index(drop=True)
    val_indices = index_df[index_df['user_id'].isin(val_user_ids)].reset_index(drop=True)
    test_indices = index_df[index_df['user_id'].isin(test_user_ids)].reset_index(drop=True)

    # Drop `user_id` column
    train_indices.drop(columns=['user_id'], inplace=True)
    val_indices.drop(columns=['user_id'], inplace=True)
    test_indices.drop(columns=['user_id'], inplace=True)
    
    return train_indices, val_indices, test_indices


def split_indices_for_infilling(df):
    results = []
    
    # Group the DataFrame by 'user_id'
    grouped = df.groupby('user_id')
    
    # Iterate over each group
    for user_id, group in grouped:
        # Get the indices of the current group
        indices = group.index
        
        # Split the indices into chunks of RAW_SEQ_LEN
        for start in range(0, len(indices), RAW_SEQ_LEN):
            end = start + RAW_SEQ_LEN
            if end <= len(indices):
                first_index = indices[start]
                last_index = indices[end - 1]
                results.append({'first_index': first_index, 'last_index': last_index, 'user_id': user_id})
    
    # Convert results list to DataFrame
    index_df = pd.DataFrame(results)

    return split_index_df_for_infilling(index_df)


def load_geolife_dataset(task):
    df = load_geolife_visit_df()
    process_space_inplace(df)
    max_duration, max_travel_time = process_time_inplace(df)

    if task == NEXT_PREDICTION:
        train_indices, val_indices, test_indices = split_indices_for_next_prediction(df)
    elif task == INFILLING:
        # train_indices, val_indices, test_indices = split_indices_for_infilling(df)
        train_indices, val_indices, test_indices = split_indices_for_next_prediction(df)
    else:
        raise NotImplementedError

    # Keep only necessary fields
    df = df[FIELDS]

    train_data = TrajGPTDataset(df, train_indices)
    val_data = TrajGPTDataset(df, val_indices)
    test_data = TrajGPTDataset(df, test_indices)

    # Number of regions
    num_regions = df['region_id'].nunique()

    # Scale of region of interest (ROI) for Space2Vec
    lambda_max = calculate_pairwise_distance_upperbound(df)

    return train_data, val_data, test_data, num_regions, lambda_max, max_duration, max_travel_time


def calculate_pairwise_distance_upperbound(df):
    """Compute scale of Region of Interest (ROI)"""
    min_x, max_x = df['x'].min(), df['x'].max()
    min_y, max_y = df['y'].min(), df['y'].max()
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
        if offsets != None:
            y += offsets.unsqueeze(1).expand(batch_size, seq_len)
        new_indices = torch.stack([x[mask], y[mask]], dim=-1).view(-1, 2)
        values = tensor[mask]
        return new_indices, values


def construct_infilling_sequence(batch, dynamic_mask):
    """Fully tensorized implementation of infilling sequence construction.

    Example:
        Original sequence: [A, B, C, D, E]
        Sequence for visit infilling: [A, BLANK, D, E, SEP, B, C, ANS]

    Args:
        batch (torch.FloatTensor): (N, L, D) tensor
        dynamic_mask (torch bool tensor): (N, L) tensor
    """
    batch_size, _, num_feats = batch.shape

    # Add unchanged & blanks
    # unchanged: ~dynamic_mask & ~is_padding
    is_padding = (batch[:, :, 0] == PAD)
    unchanged_mask = ~dynamic_mask & ~is_padding
    # blank token: first mask in each mask segments
    blank_mask = torch.concat([dynamic_mask[:, :1], dynamic_mask[:, 1:] & (~dynamic_mask)[:, :-1]], axis=1)

    # Extract values for unchanged & blanks
    cloned = batch.clone()
    cloned[blank_mask] = BLANK
    new_indices, values = get_offset_kvpair_tensorized(batch_size, cloned, unchanged_mask | blank_mask)

    # Create new tensor & fill unchanged + blanks
    seq_lens = (~is_padding).sum(axis=1)
    num_blanks = blank_mask.sum(axis=1)
    new_seq_lens = seq_lens + num_blanks * 2 + 1
    new_max_seq_len = max(new_seq_lens)

    new_batch = torch.full((batch_size, new_max_seq_len, num_feats), float(PAD))
    new_batch[new_indices[:, 0], new_indices[:, 1]] = values

    # Fill sep (+ 1 == answer index y offset)
    sep_indices = (unchanged_mask | blank_mask).sum(axis=1).long()
    new_batch[torch.arange(batch_size), sep_indices] = SEP

    # Fill target/answer part
    # Edge case: last token is masked out for prediction
    padded_dynamic_mask = torch.concat([dynamic_mask, torch.zeros(batch_size, 1).bool()], dim=-1)
    # answer token: first non-mask after each mask
    answer_mask = torch.concat([torch.zeros(batch_size, 1), (~padded_dynamic_mask)[:, 1:] & padded_dynamic_mask[:, :-1]], axis=1).bool()
    # Extract values for targets & answer tokens
    cloned = torch.concat([batch.clone(), torch.zeros((batch_size, 1, num_feats))], axis=-2)
    cloned[answer_mask] = ANS
    new_indices, values = get_offset_kvpair_tensorized(batch_size, cloned, answer_mask | padded_dynamic_mask, offsets=sep_indices+1)
    new_batch[new_indices[:, 0], new_indices[:, 1]] = values

    return new_batch


def pad_special_token_location_time_inplace(batch):
    """Replace location and timestamps of special tokens input with paddings"""
    is_special_token = (batch['region_id'] < N_SPECIAL_TOKENS)
    for key in ['arrival_time', 'departure_time', 'x', 'y']:
        batch[key][is_special_token] = PAD


def split_input_target(batch_dict):
    """Split batch dictionary into input and target dictionaries"""
    # Keeping the last token in input because we do teacher forcing for joint prediction
    input_dict = {key: batch_dict[key] for key in IN_FIELDS}
    target_dict = {key: batch_dict[key][:, 1:] for key in OUT_FIELDS}
    return input_dict, target_dict


def convert_batch_to_dict(batch):
    """Convert batch tensor to dictionary"""
    batch = {column: batch[:, :, j] for j, column in enumerate(FIELDS)}
    batch['region_id'] = batch['region_id'].long()
    return batch


def convert_batch_to_model_io(task, batch, device):
    if task == INFILLING:
        dynamic_mask = generate_dynamic_mask(batch)
        batch = construct_infilling_sequence(batch, dynamic_mask)
    batch = batch.to(device)
    batch_dict = convert_batch_to_dict(batch)
    pad_special_token_location_time_inplace(batch_dict)
    input_dict, target_dict = split_input_target(batch_dict)
    return input_dict, target_dict


def compute_after_sep_mask(region_id):
    """Compute mask for the part after SEP token"""
    sep_mask = (region_id == SEP)
    after_sep_mask = torch.cumsum(sep_mask, dim=1) > 0
    return after_sep_mask


if __name__ == "__main__":
    generate_geolife_visits()
