import numpy as np
from rdkit import Chem

ATOMS = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S']
LENS = [len(ATOMS)+1, 7, 6, 7, 5, 1]  
# atom, deg, Hs, ImpVal, Hybridization, aromatic
ATOM_MAP = {k: v for k, v in zip(ATOMS, range(0, len(ATOMS)))}
# from [0, len(ATOMS)+1]
DEGREE_MAP = {k: v for k, v in
              zip(range(LENS[1]), range(sum(LENS[:1]), sum(LENS[:2])))}

NUM_H_MAP = {k: v for k, v in
             zip(range(LENS[2]), range(sum(LENS[:2]), sum(LENS[:3])))}

IMPVAL_MAP = {k: v for k, v in
              zip(range(LENS[3]), range(sum(LENS[:3]), sum(LENS[:4])))}

HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]

assert len(HYBRIDIZATION) == LENS[-2]
UNKOWN_HYBRIDIZATION = sum(LENS[:5])-1
# TODO: this is probably a bug in the original code,
#       as it maps the unkown like "S" to SP3D2.
#       However, changing this would break the back-compatibility.

HYBRIDIZ_MAP = {k: v for k, v in
                zip(HYBRIDIZATION, range(sum(LENS[:4]), sum(LENS[:5])))}

BONDS=[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
       Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_MAP = {k: v for k, v in zip(BONDS, range(len(BONDS)))}
# TODO: this is probably a bug in the original code: 
#       no out-of-class encoding
#       However, changing this would break the back-compatibility.
#       UNKOWN_BOND = 3  # same as aromatic to circumvent crash

NUM_ATOM_FEATURES = sum(LENS)
ATOM_ENCODE_TEMPLATE = np.zeros(NUM_ATOM_FEATURES, dtype='int8')
NUM_BOND_FEATURES = len(BONDS)+2  # conjugate, inRing
BOND_ENCODE_TEMPLATE = np.zeros(NUM_BOND_FEATURES, dtype='int8')

#print(sum(LENS))
#print(ATOM_MAP)
#print(DEGREE_MAP)
#print(NUM_H_MAP)
#print(IMPVAL_MAP)
#print(HYBRIDIZ_MAP)
#print(UNKOWN_HYBRIDIZATION)

def atom_features(atom):
    #res = np.copy(ATOM_ENCODE_TEMPLATE)
    res = np.zeros(NUM_ATOM_FEATURES, dtype='int8')
    res[ATOM_MAP.get(atom.GetSymbol(), len(ATOMS))] = 1
    res[DEGREE_MAP[atom.GetDegree()]] = 1
    res[NUM_H_MAP[atom.GetTotalNumHs()]] = 1
    res[IMPVAL_MAP[atom.GetImplicitValence()]] = 1
    res[HYBRIDIZ_MAP.get(atom.GetHybridization(), UNKOWN_HYBRIDIZATION)] = 1
    res[-1] = atom.GetIsAromatic()
    return res


def atom_position(atom, conformer):
    return conformer.GetPositions()[atom.GetIndex()]

def bond_features(bond):
    #res = np.copy(BOND_ENCODE_TEMPLATE)
    res = np.zeros(NUM_BOND_FEATURES, dtype='int8')
    bt = bond.GetBondType()
    if bt in BOND_MAP.keys():
        res[BOND_MAP[bt]] = 1
    res[-2] = bond.GetIsConjugated()
    res[-1] = bond.IsInRing()
    return res

def num_atom_features():
    return NUM_ATOM_FEATURES

def num_bond_features():
    return NUM_BOND_FEATURES
