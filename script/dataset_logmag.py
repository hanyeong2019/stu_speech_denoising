import os, numpy as np, torch
from torch.utils.data import Dataset

class LogMagDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, seg=128):
        self.seg = seg; self.pairs=[]
        for f in os.listdir(noisy_dir):
            if f.endswith('_spec.npy') and os.path.exists(os.path.join(clean_dir,f)):
                self.pairs.append((os.path.join(noisy_dir,f), os.path.join(clean_dir,f)))

    def __len__(self): return len(self.pairs)

    def _load(self, p): return np.log1p(np.load(p))        # log(1+mag)
    def __getitem__(self, idx):
        n,c = self._load(self.pairs[idx][0]), self._load(self.pairs[idx][1])
        T = n.shape[1]
        if T>=self.seg:
            s=np.random.randint(0,T-self.seg+1); n,c=n[:,s:s+self.seg],c[:,s:s+self.seg]
        else:
            pad=self.seg-T; n=np.pad(n,((0,0),(0,pad))); c=np.pad(c,((0,0),(0,pad)))
        return torch.tensor(n[None],dtype=torch.float32), torch.tensor(c[None],dtype=torch.float32)
