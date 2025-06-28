import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat

file_name_dict = {
    'RML2018.10a': 'RML2018.10a.mat',
    'RML2016.10a': 'RML2016.10a.mat',
    'RML2016.10b': 'RML2016.10b.mat',
    'RML2016.10c': 'RML2016.10c.mat',
}

def single_sgn_norm(sgn: np.ndarray, normtype: str = 'maxmin'):
    if normtype == 'maxmin':
        sgn = (sgn - sgn.min()) / (sgn.max() - sgn.min())
    elif normtype == 'maxmin-1':
        sgn = (2*sgn - sgn.min() - sgn.max()) / (sgn.max() - sgn.min())
    else:
        sgn = sgn

    return sgn


def sgn_norm(sgn: np.ndarray):
    # normalize per channel
    normalized_sgn = np.zeros_like(sgn)

    for i in range(sgn.size(0)):
        iq = sgn[i]
        iq = single_sgn_norm(iq, "maxmin")
        normalized_sgn[i, :, :] = iq

    return normalized_sgn

class RMLDataset(Dataset):
    def __init__(self, data_path: str, file_name: str):
        super().__init__()
        
        # load the mat file and the data
        assert file_name in file_name_dict.keys(), f"File name {file_name} not found in {file_name_dict.keys()}"
        data = loadmat(os.path.join(data_path, file_name_dict[file_name]))
        self.data = torch.from_numpy(data['data']).float()

        # normalize the data
        self.data = sgn_norm(self.data)

        # get labels and snr
        if len(data['label'].shape) == 1:
            self.label = torch.from_numpy(data['label']).long()
        else:
            self.label = torch.from_numpy(data['label']).squeeze().t().long()
        if len(data['snr'].shape) == 1:
            self.snr = torch.from_numpy(data['snr']).t().float()
        else:
            self.snr = torch.from_numpy(data['snr']).squeeze().t().float()

        # get all the snrs
        self.snrs = np.unique(self.snr)
        self.class_num = len(np.unique(self.label))
        self.sample_num = self.data.shape[0]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index], self.snr[index]
    
if __name__ == '__main__':
    dataset = RMLDataset('/Users/chanoch/Documents/project/causal-moe/data', 'RML2016.10a')
    print(dataset.data[22000].shape, dataset.label[22000], dataset.snr[22000])