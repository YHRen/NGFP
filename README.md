# PyTorch Implementation of Neural Graph Fingerprint
forked from: https://github.com/XuhanLiu/NGFP

## Required Packages:
PyTorch, RDkit, tqdm, numpy

```
> conda install -c conda-forge rdkit
```

## Using NFP for similarity measure

### DEMO on the last target (6vww_pocket8)
[Spearman correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)

Demo shows top-20 most similar smile strings (by absolute score difference)
`rho NFP: 0.3911501585258828, rho CFP: -0.17940844415167148`

| smiles                                              |      nfp |   nfp rank |      cfp |   cfp rank |   |Δscr| |
|-----------------------------------------------------|----------|------------|----------|------------|----------|
| Cc1[nH]c2ccccc2c1C(c1cccnc1)N1CCCCC1                | 1        |          0 | 1        |        0   | 0        |
| O=C(CSc1n[nH]c(=O)n1CCc1ccccc1)c1ccc(O)cc1O         | 0.614199 |       2409 | 0.347826 |     4739.5 | 0.206902 |
| NC(=O)c1ccc(CNc2ccc3nnc(-c4ccccc4F)n3n2)cc1         | 0.575496 |       4154 | 0.377778 |     2527.5 | 0.546812 |
| O=c1[nH]c2ccc(Nc3ncnc4sc5c(c34)CCC5)cc2[nH]1        | 0.495755 |       8081 | 0.333333 |     5833.5 | 0.680997 |
| Cc1cc(C)c(-c2n[nH]c3c2C(c2ccccc2F)N(CCO)C3=O)c(O)c1 | 0.757751 |         67 | 0.365591 |     3386   | 0.716469 |
| O=C(NCCc1nc(-c2ccccc2)n[nH]1)c1cc[nH]n1             | 0.505144 |       7697 | 0.303371 |     7838.5 | 0.783464 |
| CCc1cccc2c(C(=O)N3CCCC(n4cccn4)C3)c[nH]c12          | 0.798882 |         12 | 0.413793 |      799.5 | 0.813368 |
| CCC(CNC(=O)Cn1ccc(=O)[nH]c1=O)N1CCc2ccccc2C1        | 0.625723 |       1965 | 0.377778 |     2527.5 | 0.83401  |
| CC(c1cccs1)N(C)C(=O)c1ccc2[nH]nnc2c1                | 0.69434  |        432 | 0.313953 |     7190.5 | 0.85137  |
| Cn1nc(CCNC(=O)N[C@H]2CC[C@H](O)CC2)c2ccccc21        | 0.5884   |       3502 | 0.337079 |     5528.5 | 0.926418 |
| NC(=O)CC1(O)CCCN(C(=O)c2c[nH]c(C(F)(F)F)c2)C1       | 0.761506 |         63 | 0.378049 |     2493   | 0.959059 |
| c1ccc(-c2n[nH]cc2CNC2CCCN(c3cccnn3)C2)cc1           | 0.691384 |        474 | 0.423529 |      537.5 | 1.03424  |
| Cc1cc(NC(=O)NCc2cccnc2)ccn1                         | 0.494093 |       8144 | 0.365854 |     3345   | 1.04892  |
| O=C(NCc1n[nH]c(=O)[nH]1)C1(c2ccc(Cl)cc2)CC1         | 0.670302 |        752 | 0.259259 |     9407   | 1.05528  |
| COc1ccc2[nH]c(SC(C)c3nc4ccccc4c(=O)[nH]3)nc2c1      | 0.606516 |       2709 | 0.377778 |     2527.5 | 1.06378  |
| CCc1oc2ccccc2c1CC(=O)N1CCC(c2nnc[nH]2)C1            | 0.771068 |         48 | 0.406977 |     1034.5 | 1.06719  |
| O=C(Cn1[nH]cc2c(=O)ncnc1-2)NC1(c2ccccc2)CCC1        | 0.772531 |         45 | 0.333333 |     5833.5 | 1.09636  |
| Cc1nc(Nc2ccc(C(N)=O)cc2)c2c(-c3ccccc3)csc2n1        | 0.613946 |       2425 | 0.356322 |     4064.5 | 1.15027  |
| CC(C)N(Cc1nc2c(cnn2C)c(=O)[nH]1)Cc1cccs1            | 0.632117 |       1730 | 0.233333 |     9753.5 | 1.18062  |
| Cc1ncc(COP(=O)(O)O)c(CNc2co[nH]c2=O)c1O             | 0.524038 |       6853 | 0.345238 |     4890   | 1.32263  |


### Spearman rank correlation (ρ)

| pocket         |    nfp ρ |      cfp ρ |
|----------------|----------|------------|
| 6vww_pocket108 | 0.433887 | -0.0220411 |
| 6vww_pocket9   | 0.28962  | -0.0810846 |
| 6vww_pocket100 | 0.419817 | -0.071672  |
| 6vww_pocket22  | 0.510678 | -0.141086  |
| 6vww_pocket154 | 0.667301 | -0.206292  |
| 6vww_pocket18  | 0.382344 | -0.103151  |
| 6vww_pocket57  | 0.45416  | -0.0771552 |
| 6vww_pocket143 | 0.411739 | -0.0453596 |
| 6vww_pocket3   | 0.288429 | -0.278291  |
| 6vww_pocket157 | 0.512664 | -0.279959  |
| 6vww_pocket37  | 0.625352 | -0.195684  |
| 6vww_pocket1   | 0.383276 | -0.19282   |
| 6vww_pocket13  | 0.202461 | -0.081619  |
| 6vww_pocket6   | 0.385074 | -0.128674  |
| 6vww_pocket17  | 0.458592 | -0.0679879 |
| 6vww_pocket130 | 0.485588 | -0.154715  |
| 6vww_pocket23  | 0.504353 | -0.101424  |
| 6vww_pocket71  | 0.265327 | -0.091839  |
| 6vww_pocket62  | 0.469667 | -0.0329453 |
| 6vww_pocket135 | 0.653972 | -0.0853283 |
| 6vww_pocket11  | 0.526309 | -0.103178  |
| 6vww_pocket156 | 0.554091 | -0.288988  |
| 6vww_pocket8   | 0.39115  | -0.179408  |
nfp avg: 0.44677611521262073, cfp avg: -0.13090014039680933
## MultiTask vs SingleTask accross various subsamples

![6vww fig](./figs/6vww.png)

The subsets of data are randomly selected with sizes of 5k, 10k and 20k.
All experiments are using the same fixed selections. 
Split of train, test, validation is fixed.
Averaged over 3 Runs. 

|settings    | avg. mse
-------------|----------
|single5k    | 0.844739
|single10k   | 0.692700
|single20k   | 0.660374
|multi5k     | 0.839220
|multi10k    | 0.737363
|multi20k    | 0.709497

## Examples: 

### Generate Neural Fingerprint (NFP) using a trained model

```python
python examples/generate_nfp.py --datafile <datafile.smi> \
                                --model <saved_trained_model> \
                                --output <output_nfp.npy> 
python examples/generate_nfp.py --datafile ./dataset/zinc/zinc_sample.smi \
                                --model ./output/best_efficacy.pkl.pkg \
                                --output ./output/example_nfp_output.npy
```

Each line in `<datafile.smi>` contains a smile string and additional
information. We assume the first column is the smile strings and columns are
space (or tab) separated. If not the case, one can pass the delimiter and
column index of the smile string as `--delimiter "," --column_index 2` for
example. (see function `line_parser()` in `generate_nfp.py` for more details.

Different from fingerprints as bit vectors from the Morgan algorithm, the NFP
is represented by a vector of non-negative real values.  The length of NFP is
defined by the trained model's hidden dimension (128 in the example). To change
the NFP length, One need to redefine a NFP network and re-train the model. (see
`reproduce_main_results.py` for more details.)

### Compute continuous Tanimoto Similarity

```python
> python examples/compute_tanimoto.py -h
```
Compute the continuous Tanimoto similarity, defined in the NFP paper:

![\sum_i \min(X_i, Y_i) / \sum_i \max(X_i, Y_i)](https://render.githubusercontent.com/render/math?math=%5Csum_i%20%5Cmin(X_i%2C%20Y_i)%20%2F%20%5Csum_i%20%5Cmax(X_i%2C%20Y_i))

The function `tanimoto_similarity(x,y)` is defined in `NeuralGraph/util.py`. It takes
two variables `x` and `y`: `x` must be a single fingerprint of length `L`, and `y` can be
either one fingerprint `(L,)` or an array of `M` fingerprints, `(M,L)`.

Here is an example output from the fingerprints we generated in the previous example.

```python
> python examples/compute_tanimoto.py --datafile dataset/zinc/zinc_sample.smi \
                                      --fingerprint output/example_nfp_output.npy \
                                      --top_k 10 --anchor_smile_idx 15

top-10 similar smiles to C[C@H](CCc1ccccc1)[NH2+]C[C@H](c2ccc(c(c2)C(=O)N)[O-])O
smiles                                                       score
--------------------------------------------------------  --------
C[C@H](CCc1ccccc1)[NH2+]C[C@@H](c2ccc(c(c2)C(=O)N)[O-])O  1
COc1ccc(c(c1)[O-])C(=O)NC[C@@H]2c3ccccc3CCO2              0.692871
COc1ccc(c(c1)[O-])C(=O)NC[C@H]2c3ccccc3CCO2               0.692871
Cc1cc(nc2c1cccc2)N3C[C@@H](O[C@H](C3)C)C                  0.6875
Cc1cc(nc2c1cccc2)N3C[C@H](O[C@@H](C3)C)C                  0.6875
c1ccc2c(c1)CC(=C2)N3CCN(CC3)c4cccc(c4)C(F)(F)F            0.644531
Cc1cc(ccc1OC)CCCC(=O)N/N=C/c2ccccc2[O-]                   0.64209
COc1ccc(c(c1)[O-])C(=O)Cc2cnn(c2)c3ccccc3                 0.634766
CCC(=O)c1ccc2c(c1)N(c3ccccc3S2)C[C@H](C)N(C)C             0.634766
c1ccc(cc1)COCc2cnc([nH]c2=O)[O-]                          0.633789
```


## Calculate similarity between two SMILE strings
`smile_similarity.py` takes two SMILE strings, compute their fingerprints and calculate the similarity.
Two fingerprinting methods are implemented: "morgan" and "nfp" (neural fingerprint)
If a model pkg file is not provided, the "nfp" will uses large random weights as described in the original paper.
The similarity is defined as one minus continuous Tanimoto distance.

Here is an example:

```bash
#!/bin/bash
s1="C1OC(O)C(O)C(O)C1O"
s2="CC(C)=CCCC(C)=CC(=O)"
python smile_similarity.py $s1 $s2 -m morgan
python smile_similarity.py $s1 $s2 -m nfp
python smile_similarity.py $s1 $s2 -m nfp --model './output/best_delaney.pkl.pkg'
```

## Reproduce results in the original paper
Measured in mean squared error (lower the better)

| Dataset             | Solubility    | Drug Efficacy  | Photovoltaic
| :----------         | ------------: | -------------: | -----------:
| This repo (NFP+MLP) | 0.34(0.02)    | 1.07(0.10)     | 1.08(0.06)
| NGF Paper           | 0.52(0.07)    | 1.16(0.03)     | 1.43(0.09)
| This repo (CFP+MLP) | 1.35(0.18)    | 1.13(0.03)     | 1.84(0.10)
| NGF Paper           | 1.40(0.13)    | 1.36(0.10)     | 2.00(0.09)

To reproducing these results:
```
python reproduce_main_results.py <experiment_name> <method_name>
```
where  `<experiment_name>` should be one of `["solubility", "drug_efficacy",
"photovoltaic"]`, and `<method_name>` is either `["morgan", "nfp"]`.


# Convolutional Neural Graph Fingerprint
PyTorch-based Neural Graph Fingerprint for Organic Molecule Representations

This repository is an implementation of [Convolutional Networks on Graphs for Learning Molecular Fingerprints][NGF-paper] in PyTorch.

It includes a preprocessing function to convert molecules in smiles representation
into molecule tensors.

## Related work

There are several implementations of this paper publicly available:
 - by [HIPS][1] using autograd
 - by [debbiemarkslab][2] using theano
 - by [GUR9000] [3] using keras
 - by [ericmjl][4] using autograd
 - by [DeepChem][5] using tensorflow
 - by [keiserlab][6] using Keras

The closest implementation is the implementation by GUR9000 and keiserlab in Keras. However this
repository represents moleculs in a fundamentally different way. The consequences
are described in the sections below.

## Molecule Representation

### Atom, bond and edge tensors
This codebase uses tensor matrices to represent molecules. Each molecule is
described by a combination of the following three tensors:

   - **atom matrix**, size: `(max_atoms, num_atom_features)`
   	 This matrix defines the atom features.

     Each column in the atom matrix represents the feature vector for the atom at
     the index of that column.

   - **edge matrix**, size: `(max_atoms, max_degree)`
     This matrix defines the connectivity between atoms.

     Each column in the edge matrix represent the neighbours of an atom. The
     neighbours are encoded by an integer representing the index of their feature
     vector in the atom matrix.

     As atoms can have a variable number of neighbours, not all rows will have a
     neighbour index defined. These entries are filled with the masking value of
     `-1`. (This explicit edge matrix masking value is important for the layers
     to work)

   - **bond tensor** size: `(max_atoms, max_degree, num_bond_features)`
   	 This matrix defines the atom features.

   	 The first two dimensions of this tensor represent the bonds defined in the
   	 edge tensor. The column in the bond tensor at the position of the bond index
   	 in the edge tensor defines the features of that bond.

   	 Bonds that are unused are masked with 0 vectors.


### Batch representations

 This codes deals with molecules in batches. An extra dimension is added to all
 of the three tensors at the first index. Their respective sizes become:

 - **atom matrix**, size: `(num_molecules, max_atoms, num_atom_features)`
 - **edge matrix**, size: `(num_molecules, max_atoms, max_degree)`
 - **bond tensor** size: `(num_molecules, max_atoms, max_degree, num_bond_features)`

As molecules have different numbers of atoms, max_atoms needs to be defined for
the entire dataset. Unused atom columns are masked by 0 vectors.


## Dependencies
- [**RDKit**](http://www.rdkit.org/) This dependency is necessary to convert molecules into tensor
representatins, once this step is conducted, the new data can be stored, and RDkit
is no longer a dependency.
- [**PyTorch**](https://PyTorch.org/) Requires PyTorch >= 1.0
- [**NumPy**](http://www.numpy.org/) Requires Numpy >= 0.19
- [**Pandas**](http://www.pandas.org) Optional for examples

## Acknowledgements
- Implementation is based on [Duvenaud et al., 2015][NGF-paper].
- Feature extraction scripts were implemented from [the original implementation][1]
- Data preprocessing scripts were rewritten from [keiserlab][3]
- Graphpool layer adopted from [Han, et al., 2016][DeepChem-paper]

[NGF-paper]: https://arxiv.org/abs/1509.09292
[DeepChem-paper]:https://arxiv.org/abs/1611.03199
[keiserlab]: //http://www.keiserlab.org/
[1]: https://github.com/HIPS/neural-fingerprint
[2]: https://github.com/debbiemarkslab/neural-fingerprint-theano
[3]: https://github.com/GUR9000/KerasNeuralFingerprint
[4]: https://github.com/ericmjl/graph-fingerprint
[5]: https://github.com/deepchem/deepchem
[6]: https://github.com/keiserlab/keras-neural-graph-fingerprint




<!---
### Grid search activations for GraphConvNet (gcn) and GraphOutput (gop)
Gridsaerch of 
```python
gcn_act = ['sigmoid', 'relu', 'tanh']
gop_act = ['sigmoid', 'tanh', 'softmax']
large_weights = [(-1e7, 1e7), (0, 1e7), (-1e3, 1e3), (-10, 10)]
max_degs = [1, 6]
```
|params                                                             |  correlation
|------------------------------------------------------------------ |-------------
|gcn-sigmoid_gop-softmax_weights-(-1000.0, 1000.0)_radius-1         |    0.716294
|gcn-sigmoid_gop-softmax_weights-(-10000000.0, 10000000.0)_radius-1 |    0.679691
|gcn-sigmoid_gop-softmax_weights-(0, 10000000.0)_radius-1           |    0.642413
|gcn-sigmoid_gop-tanh_weights-(-10, 10)_radius-1                    |    0.618465
|gcn-sigmoid_gop-softmax_weights-(-10, 10)_radius-1                 |    0.612766
|gcn-sigmoid_gop-sigmoid_weights-(-10000000.0, 10000000.0)_radius-1 |    0.55004
|gcn-relu_gop-sigmoid_weights-(-10000000.0, 10000000.0)_radius-1    |    0.536428
|gcn-relu_gop-sigmoid_weights-(-1000.0, 1000.0)_radius-1            |    0.532326
|gcn-relu_gop-sigmoid_weights-(-10, 10)_radius-1                    |    0.531631
|gcn-sigmoid_gop-sigmoid_weights-(-10, 10)_radius-1                 |    0.53001
|gcn-sigmoid_gop-sigmoid_weights-(-1000.0, 1000.0)_radius-1         |    0.529918
|gcn-relu_gop-tanh_weights-(-10000000.0, 10000000.0)_radius-6       |    0.479653
|gcn-relu_gop-sigmoid_weights-(-1000.0, 1000.0)_radius-6            |    0.475187
|gcn-sigmoid_gop-softmax_weights-(-10000000.0, 10000000.0)_radius-6 |    0.47381
|gcn-relu_gop-sigmoid_weights-(-10000000.0, 10000000.0)_radius-6    |    0.458613
|gcn-sigmoid_gop-softmax_weights-(-10, 10)_radius-6                 |    0.457012
|gcn-relu_gop-sigmoid_weights-(-10, 10)_radius-6                    |    0.454613
|gcn-sigmoid_gop-sigmoid_weights-(-10, 10)_radius-6                 |    0.418538
|gcn-sigmoid_gop-sigmoid_weights-(-10000000.0, 10000000.0)_radius-6 |    0.406702
|gcn-sigmoid_gop-sigmoid_weights-(-1000.0, 1000.0)_radius-6         |    0.375891
|gcn-sigmoid_gop-tanh_weights-(-10000000.0, 10000000.0)_radius-6    |    0.372162
|gcn-sigmoid_gop-softmax_weights-(-1000.0, 1000.0)_radius-6         |    0.352566
|gcn-sigmoid_gop-softmax_weights-(0, 10000000.0)_radius-6           |    0.311116
|gcn-sigmoid_gop-sigmoid_weights-(0, 10000000.0)_radius-1           |    0.295567
|gcn-sigmoid_gop-sigmoid_weights-(0, 10000000.0)_radius-6           |    0.295567
|gcn-sigmoid_gop-tanh_weights-(0, 10000000.0)_radius-1              |    0.295567
|gcn-sigmoid_gop-tanh_weights-(0, 10000000.0)_radius-6              |    0.295567
|gcn-relu_gop-sigmoid_weights-(0, 10000000.0)_radius-1              |    0.295567
|gcn-relu_gop-sigmoid_weights-(0, 10000000.0)_radius-6              |    0.295567
|gcn-relu_gop-tanh_weights-(0, 10000000.0)_radius-1                 |    0.295567
|gcn-relu_gop-tanh_weights-(0, 10000000.0)_radius-6                 |    0.295567
|gcn-sigmoid_gop-tanh_weights-(-10, 10)_radius-6                    |    0.261334
|gcn-sigmoid_gop-tanh_weights-(-1000.0, 1000.0)_radius-6            |    0.2468
|gcn-sigmoid_gop-tanh_weights-(-1000.0, 1000.0)_radius-1            |    0.194475
|gcn-relu_gop-tanh_weights-(-10000000.0, 10000000.0)_radius-1       |    0.139468
|gcn-sigmoid_gop-tanh_weights-(-10000000.0, 10000000.0)_radius-1    |   -0.095261
|gcn-relu_gop-tanh_weights-(-10, 10)_radius-6                       |  nan
|gcn-relu_gop-softmax_weights-(-10, 10)_radius-1                    |    0.686585
|gcn-tanh_gop-softmax_weights-(-10000000.0, 10000000.0)_radius-1    |    0.665152
|gcn-tanh_gop-softmax_weights-(-10, 10)_radius-1                    |    0.665107
|gcn-relu_gop-softmax_weights-(0, 10000000.0)_radius-1              |    0.657383
|gcn-tanh_gop-softmax_weights-(-1000.0, 1000.0)_radius-1            |    0.629601
|gcn-tanh_gop-softmax_weights-(0, 10000000.0)_radius-1              |    0.604808
|gcn-relu_gop-softmax_weights-(-1000.0, 1000.0)_radius-6            |    0.581197
|gcn-relu_gop-softmax_weights-(-1000.0, 1000.0)_radius-1            |    0.572924
|gcn-relu_gop-tanh_weights-(-1000.0, 1000.0)_radius-6               |    0.565224
|gcn-relu_gop-softmax_weights-(-10000000.0, 10000000.0)_radius-1    |    0.562611
|gcn-relu_gop-tanh_weights-(-1000.0, 1000.0)_radius-1               |    0.560201
|gcn-relu_gop-softmax_weights-(-10, 10)_radius-6                    |    0.550639
|gcn-tanh_gop-softmax_weights-(0, 10000000.0)_radius-6              |    0.539548
|gcn-tanh_gop-sigmoid_weights-(-10, 10)_radius-1                    |    0.52877
|gcn-tanh_gop-sigmoid_weights-(-1000.0, 1000.0)_radius-1            |    0.525169
|gcn-tanh_gop-sigmoid_weights-(-10000000.0, 10000000.0)_radius-1    |    0.52363
|gcn-tanh_gop-sigmoid_weights-(-10, 10)_radius-6                    |    0.438762
|gcn-tanh_gop-sigmoid_weights-(-1000.0, 1000.0)_radius-6            |    0.43075
|gcn-tanh_gop-softmax_weights-(-10000000.0, 10000000.0)_radius-6    |    0.430058
|gcn-tanh_gop-softmax_weights-(-10, 10)_radius-6                    |    0.424098
|gcn-tanh_gop-sigmoid_weights-(-10000000.0, 10000000.0)_radius-6    |    0.421994
|gcn-relu_gop-softmax_weights-(-10000000.0, 10000000.0)_radius-6    |    0.363453
|gcn-tanh_gop-softmax_weights-(-1000.0, 1000.0)_radius-6            |    0.345484
|gcn-tanh_gop-tanh_weights-(-1000.0, 1000.0)_radius-6               |    0.340882
|gcn-tanh_gop-tanh_weights-(-1000.0, 1000.0)_radius-1               |    0.320849
|gcn-relu_gop-softmax_weights-(0, 10000000.0)_radius-6              |    0.295567
|gcn-tanh_gop-sigmoid_weights-(0, 10000000.0)_radius-1              |    0.295567
|gcn-tanh_gop-sigmoid_weights-(0, 10000000.0)_radius-6              |    0.295567
|gcn-tanh_gop-tanh_weights-(0, 10000000.0)_radius-1                 |    0.295567
|gcn-tanh_gop-tanh_weights-(0, 10000000.0)_radius-6                 |    0.295567
|gcn-tanh_gop-tanh_weights-(-10000000.0, 10000000.0)_radius-6       |    0.240071
|gcn-tanh_gop-tanh_weights-(-10, 10)_radius-1                       |    0.229624
|gcn-tanh_gop-tanh_weights-(-10000000.0, 10000000.0)_radius-1       |    0.209503
|gcn-relu_gop-tanh_weights-(-10, 10)_radius-1                       |    0.0741423
|gcn-tanh_gop-tanh_weights-(-10, 10)_radius-6                       |   -0.0714465
--->
