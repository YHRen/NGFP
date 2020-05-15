import numpy as np
from rdkit import Chem
from . import feature_static as feature  # more efficient implementation
from .feature_static import NUM_ATOM_FEATURES, NUM_BOND_FEATURES
from .preprocessing import tensorise_smiles as tensorise_smiles_slow
from tqdm import tqdm
from timeit import default_timer as timer
import torch


def tensorise_one_molecule(mol, n, max_degree=6):
    # Make sure the input sml is valid
    # n := max num of atoms
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    atom_tensor = np.zeros((n, NUM_ATOM_FEATURES), dtype='int8')
    bond_tensor = np.zeros((n, max_degree, NUM_BOND_FEATURES), dtype='int8')
    edge_tensor = -np.ones((n, max_degree),dtype='int8')
    nbr_cnt = np.zeros(n, dtype='int8')

    for atom_ix, atom in enumerate(atoms):
        atom_tensor[atom.GetIdx(), :] = feature.atom_features(atom)
    
    for bond in bonds:
        a1_ix = bond.GetBeginAtom().GetIdx()
        a2_ix = bond.GetEndAtom().GetIdx()
        
        a1_neigh = nbr_cnt[a1_ix]
        a2_neigh = nbr_cnt[a2_ix]

        # If max_degree is exceeded, report error 
        assert max(a1_neigh, a2_neigh) + 1 <= max_degree, \
            "too many neighours ({0}) in molecule: \
            {1}".format(max(a1_neigh, a2_neigh) + 1, sml)
        
        edge_tensor[a1_ix, a1_neigh] = a2_ix
        edge_tensor[a2_ix, a2_neigh] = a1_ix
        bond_tensor[a1_ix, a1_neigh] = bond_tensor[a2_ix, a2_neigh] = feature.bond_features(bond)
        nbr_cnt[a1_ix] += 1
        nbr_cnt[a2_ix] += 1

    return atom_tensor, bond_tensor, edge_tensor


def get_mol_wrapper(sml):
    """ to circumvent the pikle issue of boost python function in  rdkit"""
    return Chem.MolFromSmiles(sml)


def tensorise_smiles(smiles, max_degree=5, max_atoms=None, use_tqdm=False,
                     worker_pool=None):
    """Takes a list of smiles and turns the graphs in tensor representation.
    # Arguments:
        smiles: a list (or iterable) of smiles representations
        max_atoms: the maximum number of atoms per molecule (to which all
            molecules will be padded), use `None` for auto
        max_degree: max_atoms: the maximum number of neigbour per atom that each
            molecule can have (to which all molecules will be padded), use `None`
            for auto
        **NOTE**: It is not recommended to set max_degree to `None`/auto when
            using `NeuralGraph` layers. Max_degree determines the number of
            trainable parameters and is essentially a hyperparameter.
            While models can be rebuilt using different `max_atoms`, they cannot
            be rebuild for different values of `max_degree`, as the architecture
            will be different.
            For organic molecules `max_degree=5` is a good value (Duvenaud et. al, 2015)
    # Returns:
        atoms: np.array, An atom feature np.array of size `(molecules, max_atoms, atom_features)`
        bonds: np.array, A bonds np.array of size `(molecules, max_atoms, max_neighbours)`
        edges: np.array, A connectivity array of size `(molecules, max_atoms, max_neighbours, bond_features)`
    """
    if not worker_pool:  # call the original slow version
        return tensorise_smiles_slow(smiles, max_degree, max_atoms, use_tqdm)

    mols = worker_pool.map(get_mol_wrapper, smiles)
    max_atoms = [max(m.GetNumAtoms() for m in mols)]*len(mols)
    res = worker_pool.starmap(tensorise_one_molecule, zip(mols, max_atoms))
    atom_tensor = np.asarray([x[0] for x in res])
    bond_tensor = np.asarray([x[1] for x in res])
    edge_tensor = np.asarray([x[2] for x in res])

    return torch.from_numpy(atom_tensor,).float(), \
           torch.from_numpy(bond_tensor,).float(), \
           torch.from_numpy(edge_tensor,).long()
