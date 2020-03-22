# For the reproducing the result of drug efficacy

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from NeuralGraph.dataset import MolData
from NeuralGraph.model import QSAR
import torch.nn as nn
import pandas as pd
import numpy as np


def split_train_valid(n, p=0.8, seed=None):
	if seed: np.random.seed(seed)
	idx = np.arange(n)
	np.random.shuffle(idx)
	s = int(n*p)
	return idx[:s], idx[s:]

def split_train_valid_test(n, p=0.8, v=0.1, seed=None):
	if seed: np.random.seed(seed)
	idx = np.arange(n)
	np.random.shuffle(idx)
	s = int(n*p)
	t = int(n*v)
	# train, valid, test
	return idx[:s], idx[s:(s+t)], idx[(s+t):]

def normalize_array(A):
	mean, std = np.mean(A), np.std(A)
	def norm_func(X): return (X-mean) / std
	def restore_func(X): return X * std + mean
	return norm_func, restore_func

def load_efficacy(data_file):
	df = pd.read_csv(data_file)
	input = df['smiles']
	target = df['activity'].values
	return input, target
	target, restore = normalize_array(target)
	data = MolData(df['smiles'], target)
	return data, target, restore

def mse(x, y):
	return ((x-y)**2).mean()


def main():
	BSZ = 128
	RUNS = 5
	N_EPOCH = 100
	LR = 1e-3
	OUTPUT = './output/best_efficacy.pkl'
	DATAFILE = Path('./dataset/efficacy/malaria-processed.csv')


	res = []
	for _ in range(RUNS):
		input_data, target = load_efficacy(DATAFILE)
		train_idx, valid_idx, test_idx = split_train_valid_test(len(target), seed=None)
		norm_func, restore_func = normalize_array( \
			np.concatenate([target[train_idx], target[valid_idx]], axis=0))
		target = norm_func(target)
		data = MolData(input_data, target)
		train_loader = DataLoader(Subset(data, train_idx), batch_size=BSZ, \
								  shuffle=True, drop_last=True)
		valid_loader = DataLoader(Subset(data, valid_idx), batch_size=BSZ, \
								  shuffle=False)
		test_loader = DataLoader(Subset(data, test_idx), batch_size=BSZ, \
							  shuffle=False)
		net = QSAR(hid_dim=128, n_class=1)
		net = net.fit(train_loader, valid_loader, epochs=N_EPOCH, path=OUTPUT,
					  criterion=nn.MSELoss(), lr=LR)
		score = net.predict(test_loader)
		gt = restore_func(target[test_idx])
		prd = restore_func(score)
		res.append(mse(gt, prd))
		print(mse(gt,prd))

	print(np.asarray(res).mean(), np.asarray(res).std())


if __name__ == '__main__':
	main()
	