#code from vamb
import numpy as _np
import torch as _torch
_torch.manual_seed(0)

from math import log as _log
import math

from torch import nn as _nn
from torch.optim import Adam as _Adam
from torch.nn.functional import softmax as _softmax
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.dataset import TensorDataset as _TensorDataset

from DRBin.vMF import *
import DRBin
import DRBin.utils

def numpy_inplace_maskarray(array, mask):

    uints = _np.frombuffer(mask, dtype=_np.uint8)
    index = _overwrite_matrix(array, uints)
    array.resize((index, array.shape[1]), refcheck=False)
    return array

def zscore(array, axis=None, inplace=False):

    mean = array.mean(axis=axis)
    std = array.std(axis=axis)

    if axis is None:
        if std == 0:
            std = 1

    else:
        std[std == 0.0] = 1
        shape = tuple(dim if ax != axis else 1 for ax, dim in enumerate(array.shape))
        mean.shape, std.shape = shape, shape

    if inplace:
        array -= mean
        array /= std
        return None
    else:
        return (array - mean) / std
    

def make_dataloader(rpkm, tnf, batchsize=256, destroy=False, cuda=True):
    mask = tnf.sum(axis=1) != 0

    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        mask &= depthssum != 0
        depthssum = depthssum[mask]

    if destroy:
        rpkm = numpy_inplace_maskarray(rpkm, mask)
        tnf = numpy_inplace_maskarray(tnf, mask)
    else:
        rpkm = rpkm[mask].astype(_np.float32, copy=False)
        tnf = tnf[mask].astype(_np.float32, copy=False)

    if rpkm.shape[1] > 1:
        rpkm /= depthssum.reshape((-1, 1))
    else:
        zscore(rpkm, axis=0, inplace=True)

    zscore(tnf, axis=0, inplace=True)
    depthstensor = _torch.from_numpy(rpkm)
    tnftensor = _torch.from_numpy(tnf)

    # 创建dataloader
    n_workers = 4 if cuda else 1
    dataset = _TensorDataset(depthstensor, tnftensor)
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda)

    return dataloader

class vMF_VAE(_nn.Module):
    def __init__(self, nsamples, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        super(vMF_VAE, self).__init__()

        # 初始化
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = 136
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout

        self.encoderlayers = _nn.ModuleList()
        self.encodernorms = _nn.ModuleList()
        self.decoderlayers = _nn.ModuleList()
        self.decodernorms = _nn.ModuleList()

        for nin, nout in zip([self.nsamples + self.ntnf] + self.nhiddens, self.nhiddens):
            self.encoderlayers.append(_nn.Linear(nin, nout))
            self.encodernorms.append(_nn.BatchNorm1d(nout))

        self.mu = _nn.Linear(self.nhiddens[-1], self.nlatent)
        self.logsigma = _nn.Linear(self.nhiddens[-1], 1)

        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(_nn.Linear(nin, nout))
            self.decodernorms.append(_nn.BatchNorm1d(nout))

        self.outputlayer = _nn.Linear(self.nhiddens[0], self.nsamples + self.ntnf)

        self.relu = _nn.LeakyReLU()
        self.softplus = _nn.Softplus()
        self.dropoutlayer = _nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()

    def _encode(self, tensor):
        tensors = list()

        # 隐藏层
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        # Latent layers
        mu = self.mu(tensor)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        
        logsigma = self.softplus(self.logsigma(tensor)) + 1

        return mu, logsigma

    # 重采样
    def reparameterize(self, mu, logsigma):
        q_z = VonMisesFisher(mu, logsigma)
        p_z = HypersphericalUniform(self.nlatent - 1)
        
        latent = q_z.rsample()
        
        return latent, q_z, p_z

    def _decode(self, tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # 解耦合序列丰度信息和序列组成信息
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)

        if self.nsamples > 1:
            depths_out = _softmax(depths_out, dim=1)

        return depths_out, tnf_out

    def forward(self, depths, tnf):
        tensor = _torch.cat((depths, tnf), 1)
        mu, logsigma = self._encode(tensor)
        latent, q_z, p_z = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self._decode(latent)

        return depths_out, tnf_out, mu, logsigma, q_z, p_z

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, q_z, p_z):
        # 多个样本则采用交叉墒，否则采用重构损失
        if self.nsamples > 1:
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / _log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = (-q_z.entropy() + p_z.entropy()).mean()
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce * ce_weight + sse * sse_weight + kld * kld_weight

        return loss, ce, sse, kld

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_celoss = 0

        if epoch in batchsteps:
            data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for depths_in, tnf_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, mu, logsigma, q_z, p_z = self(depths_in, tnf_in)

            loss, ce, sse, kld = self.calc_loss(depths_in, depths_out, tnf_in,
                                                  tnf_out, q_z, p_z)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()
            epoch_celoss += ce.data.item()
        
        print(
                "[Epoch %d] [epoch_loss: %f] [epoch_kldloss: %f] [epoch_sse: %f] [epoch_celoss: %f]"
                % (epoch, epoch_loss, epoch_kldloss, epoch_sseloss, epoch_celoss))

        return data_loader

    def encode(self, data_loader):
        """利用VAE编码的过程
        Input: data_loader: 训练数据
        Output: A (n_contigs x n_latent) 隐向量
        """

        self.eval()

        new_data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=1,
                                      pin_memory=data_loader.pin_memory)

        depths_array, tnf_array = data_loader.dataset.tensors
        length = len(depths_array)
 
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()

                # Evaluate
                out_depths, out_tnf, mu, logsigma, q_z, p_z = self(depths, tnf)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent

    def trainmodel(self, dataloader, nepochs=500, lrate=1e-3,
                   batchsteps=[25, 75, 150, 300], modelfile=None):

        if batchsteps is None:
            batchsteps_set = set()
        else:
            batchsteps = list(batchsteps)
            last_batchsize = dataloader.batch_size * 2**len(batchsteps)
            batchsteps_set = set(batchsteps)

        ncontigs, nsamples = dataloader.dataset.tensors[0].shape
        optimizer = _Adam(self.parameters(), lr=lrate)
        
        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(dataloader, epoch, optimizer, batchsteps_set)

        return None