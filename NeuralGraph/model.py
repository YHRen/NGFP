import sys
import torch as T
from torch import nn
from .layer import GraphConv, GraphPool, GraphOutput
import numpy as np
from torch import optim
import time
from .util import dev, calc_neural_fp
from .preprocessing import tensorise_smiles

class NeuralFingerPrint(nn.Module):
    def __init__(self, hid_dim, max_degree=6,\
                 gcn_activation='relu', gop_activation='softmax'):
        super(NeuralFingerPrint, self).__init__()
        self.gcn1 = GraphConv(input_dim=43, conv_width=128,
                              max_degree=max_degree, activation=gcn_activation)
        self.gcn2 = GraphConv(input_dim=134, conv_width=128,
                              max_degree=max_degree, activation=gcn_activation)
        self.gop = GraphOutput(input_dim=134, output_dim=128,
                               activation=gop_activation)
        self.pool = GraphPool()
        self.to(dev)
    
    def forward(self, atoms, bonds, edges):
        atoms = self.gcn1(atoms, bonds, edges)
        # atoms = self.bn(atoms)
        atoms = self.pool(atoms, edges)
        atoms = self.gcn2(atoms, bonds, edges)
        # atoms = self.bn(atoms)
        atoms = self.pool(atoms, edges)
        fp = self.gop(atoms, bonds, edges)
        return fp

class MLP(nn.Module):
    def __init__(self, hid_dim, n_class):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(hid_dim, hid_dim//2),
                                 nn.ReLU(),
                                 nn.Linear(hid_dim//2, hid_dim//4),
                                 nn.ReLU(),
                                 nn.Linear(hid_dim//4, n_class))
        self.to(dev)

    def forward(self, fp):
        return self.mlp(fp).squeeze()

    def fit(self, loader_train, loader_valid, path, \
            criterion=nn.BCEWithLogitsLoss(), \
            epochs=1000, early_stop=100, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        best_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            t0 = time.time()
            for xb, yb in loader_train:
                xb, yb = xb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                y_ = self.forward(xb)
                loss = criterion(y_, yb)
                loss.backward()
                optimizer.step()
            loss_valid = self.evaluate(loader_valid, criterion=criterion)
            print('[Epoch:%d/%d] %.1fs loss_train: %f loss_valid: %f' % (
                epoch, epochs, time.time() - t0, loss.item(), loss_valid),
                  file=sys.stderr)
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (
                    best_loss, loss_valid, path + '.pkg'), file=sys.stderr)
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.',
                      file=sys.stderr)
            if early_stop is not None and epoch - last_saving > early_stop: break
        return T.load(path + '.pkg')

    def evaluate(self, loader, criterion=nn.BCEWithLogitsLoss()):
        loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            y_ = self.forward(xb)
            loss += criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            y_ = self.forward(xb)
            score.append(y_.data.cpu())
        score = T.cat(score, dim=0).numpy()
        return score

class QSAR(nn.Module):
    def __init__(self, hid_dim, n_class, max_degree=6, \
                 gcn_activation='relu', gop_activation='softmax'):
        super(QSAR, self).__init__()
        self.nfp = NeuralFingerPrint(hid_dim, max_degree=max_degree,
                 gcn_activation=gcn_activation, gop_activation=gop_activation)
        # self.bn = nn.BatchNorm2d(80)
        self.mlp = MLP(hid_dim, n_class)
        self.to(dev)

    def forward(self, atoms, bonds, edges):
        fp = self.nfp(atoms, bonds, edges)
        out = self.mlp(fp).squeeze()
        return out

    def fit(self, loader_train, loader_valid, path, \
            criterion=nn.BCEWithLogitsLoss(), \
            epochs=1000, early_stop=100, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        best_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            t0 = time.time()
            for (Ab, Bb, Eb), yb in loader_train:
                Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                y_ = self.forward(Ab, Bb, Eb)
                #ix = yb == yb
                #yb, y_ = yb[ix], y_[ix]
                loss = criterion(y_, yb)
                loss.backward()
                optimizer.step()
            loss_valid = self.evaluate(loader_valid, criterion=criterion)
            print('[Epoch:%d/%d] %.1fs loss_train: %f loss_valid: %f' % (
                epoch, epochs, time.time() - t0, loss.item(), loss_valid),
                  file=sys.stderr)
            if loss_valid < best_loss:
                T.save(self, path + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (
                    best_loss, loss_valid, path + '.pkg'),
                      file=sys.stderr)
                best_loss = loss_valid
                last_saving = epoch
            else:
                print('[Performance] loss_valid is not improved.',
                      file=sys.stderr)
            if early_stop is not None and epoch - last_saving > early_stop: break
        return T.load(path + '.pkg')

    def evaluate(self, loader, criterion=nn.BCEWithLogitsLoss()):
        loss = 0
        for (Ab, Bb, Eb), yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb)
            #ix = yb == yb
            #yb, y_ = yb[ix], y_[ix]
            loss += criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for (Ab, Bb, Eb), yb in loader:
            Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
            y_ = self.forward(Ab, Bb, Eb)
            score.append(y_.data.cpu())
        score = T.cat(score, dim=0).numpy()
        return score

    def calc_nfp(self, smiles, is_float16=True):
        """ Calculate the neural fingerprint (nfp) based on the current neural
        network.

        input:

        smiles: a list of SMILE strings of size **n**. 
        
        
        output:

            a numpy matrix of floats of dimension **n-by-hid_dim**, where
            **hid_dim** is the length of nfp.
            

        NOTE: make sure the SMILE strings can be fitted into the GPU memory.

        Example:

        net = torch.load("<saved_model_file>")
        res = []
        bsz = 256
        with open('./dataset.smiles','r') as fp:
            cnt, cache = 0, []
            for line in fp:
                if cnt < bsz:
                    cache.append(line)
                    cnt += 1
                else:
                    res.append(net.calc_nfp(cache))
                    cnt, cache = 0, []
            if len(cache) > 0:
                res.append(net.calc_nfp(cache))
        """
        tmp = tensorise_smiles(smiles)
        res = calc_neural_fp(tmp, self)
        if is_float16:
            res = np.float16(res)
        return res

