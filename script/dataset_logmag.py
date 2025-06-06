import os, numpy as np, torch
from torch.utils.data import Dataset

class LogMagDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, seg=128, mode='train'):
        self.seg = seg
        self.mode = mode  # 新增
        self.pairs = []
        for f in os.listdir(noisy_dir):
            if f.endswith('_spec.npy') and os.path.exists(os.path.join(clean_dir, f)):
                self.pairs.append((os.path.join(noisy_dir, f), os.path.join(clean_dir, f)))

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        return np.log1p(np.load(path))  # log(1 + mag)

    # def __getitem__(self, idx):
    #     n, c = self._load(self.pairs[idx][0]), self._load(self.pairs[idx][1])
    #     T = n.shape[1]
    #     if T >= self.seg:
    #         if self.mode == 'train':
    #             s = np.random.randint(0, T - self.seg + 1)
    #         else:
    #             s = 0  # 评估时从开头截取固定段
    #         n, c = n[:, s:s + self.seg], c[:, s:s + self.seg]
    #     else:
    #         pad = self.seg - T
    #         n = np.pad(n, ((0, 0), (0, pad)))
    #         c = np.pad(c, ((0, 0), (0, pad)))
    #     return torch.tensor(n[None], dtype=torch.float32), torch.tensor(c[None], dtype=torch.float32)


    def __getitem__(self, idx):
     n_path, c_path = self.pairs[idx]
     n, c = self._load(n_path), self._load(c_path)
     T = n.shape[1]
     if T >= self.seg:
        s = np.random.randint(0, T - self.seg + 1)
        n, c = n[:, s:s + self.seg], c[:, s:s + self.seg]
     else:
        pad = self.seg - T
        n = np.pad(n, ((0, 0), (0, pad)))
        c = np.pad(c, ((0, 0), (0, pad)))

     basename = os.path.basename(n_path)
     return torch.tensor(n[None], dtype=torch.float32), torch.tensor(c[None], dtype=torch.float32), basename
