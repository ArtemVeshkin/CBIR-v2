from torch.utils.data import Dataset
import os


class FolderDataset(Dataset):
    def __init__(self, data_dir, load_fn, transform_fn):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir)
        self.load_fn = load_fn
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data[idx])
        loaded = self.load_fn(file_path)
        transformed = self.transform_fn(loaded)
        return transformed

