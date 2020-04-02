import torch as T
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from . import feature
from . import preprocessing as prep
from . import util


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
    def __init__(self, smiles, labels, use_tqdm=False):
        self.max_atom = 80
        self.max_degree = 6
        self.atoms, self.bonds, self.edges = self._featurize(smiles,
                                                             use_tqdm=use_tqdm)
        self.label = T.from_numpy(labels).float()

    def _featurize(self, smiles, use_tqdm=False):
        return prep.tensorise_smiles(smiles, max_atoms=self.max_atom,
                                     max_degree=self.max_degree,
                                     use_tqdm=use_tqdm)

    def __getitem__(self, i):
        return (self.atoms[i], self.bonds[i], self.edges[i]), self.label[i]

    def __len__(self):
        return len(self.label)


class SmileData(Dataset):
    """Dataset for circular finger print"""
    def __init__(self, smiles, labels, fp_len=128, radius=6):
        self.fp_len, self.radius = fp_len, radius
        self.cfp = self._calc_cfp(smiles)
        self.label = T.from_numpy(labels).float()

    def _calc_cfp(self, smiles):
        res = []
        for s in smiles:
            res.append(util.calc_circular_fp(s, fp_len=self.fp_len,
                                             radius=self.radius))
        return T.from_numpy(np.asarray(res)).float()

    def __getitem__(self, i):
        return self.cfp[i], self.label[i]

    def __len__(self):
        return len(self.label)

