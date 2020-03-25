# PyTorch Implementation of Neural Graph Fingerprint
forked from: https://github.com/XuhanLiu/NGFP


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
python measure_smile_distance.py $s1 $s2 -m morgan
python measure_smile_distance.py $s1 $s2 -m nfp
python measure_smile_distance.py $s1 $s2 -m nfp --model './output/best_delaney.pkl.pkg'
```

## Reproduce results in the original paper
Measured in mean squared error (lower the better)

| Dataset     | Solubility   | Drug Efficacy | Photovoltaic
| :---------- | ------------:| -------------:| -----------:
| This Repo   | 0.43(0.11)   | 1.17(0.07)    | TODO
| NGF Paper   | 0.52(0.07)   | 1.16(0.03)    | 1.43(0.09)

To reproducing these results:
```
python reproduce_main_results.py <experiment_name>
```
where  `<experiment_name>` should be one of `["solubility", "drug_efficacy", "photovoltaic"]`.


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
