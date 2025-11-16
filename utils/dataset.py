import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.constants import FIELDS


class TrajGPTDataset(Dataset):
    def __init__(self, df, indices):
        self.df = df
        self.indices = indices.astype(int)
        self.df["padding_mask"] = False  # No padding within each trajectory

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        first_index, last_index = self.indices.loc[idx].values
        return self.df.loc[first_index:last_index]


def collate_fn(dfs):
    """Collate function to pad sequences in the batch."""
    padded_batch = {}
    for col in dfs[0].columns:
        value = [torch.from_numpy(df[col].values) for df in dfs]
        if col == "padding_mask":
            padded_batch[col] = pad_sequence(
                value, batch_first=True, padding_value=True
            )
        elif col in FIELDS:
            padded_batch[col] = pad_sequence(
                value, batch_first=True, padding_value=0
            )  # Pad with 0 for other features
            if col == "place_id":
                padded_batch[col] = padded_batch[col].long()
            else:
                padded_batch[col] = padded_batch[col].float()

    return padded_batch
