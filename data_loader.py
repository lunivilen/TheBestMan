from torch.utils.data import Dataset
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleDatasetLoader(Dataset):
    def __init__(self, df):
        self.data_x = torch.tensor(df.drop("TARGET", axis=1).to_numpy(),
                                   dtype=torch.float32,
                                   device=device)

        self.data_y = torch.tensor(df["TARGET"].to_numpy(),
                                   dtype=torch.float32,
                                   device=device)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index].reshape(1)

    def __len__(self):
        return len(self.data_x)
