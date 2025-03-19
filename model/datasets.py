import torch
from torch.utils.data import Dataset


class DatasetCSTS(Dataset):
    def __init__(self, file_path,tokenizer):
        self.dataset = []
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                self.dataset.append([line[1], line[3], float(line[4]) * 0.4 - 1])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return {
            "sentence1":data[0],
            "sentence2":data[1],
            "score":torch.tensor(data[2], dtype=torch.float32)
        }


    