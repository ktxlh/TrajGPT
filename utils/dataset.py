import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
from utils.constants import PAD


class TrajGPTDataset(Dataset):
    def __init__(self, df, indices, sequence_length):
        self.df = df
        self.sequence_length = sequence_length
        self.indices = indices.astype(int)

    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        first_index, last_index = self.indices.loc[idx].values
        instance = self.df.loc[first_index:last_index].values
        instance = torch.from_numpy(instance).float()

        # Pad the sequence to the maximum length
        if instance.shape[0] < self.sequence_length:
            # Pad left instead of right so the last element is the target
            instance = pad(instance, (0, 0, self.sequence_length - instance.shape[0], 0), mode='constant', value=PAD)
        return instance
