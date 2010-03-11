#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

import bolt
import io
import model

ftrain =  "/media/disk-3/data/corpora/rcv1/train.npy"
ftest = "/media/disk-3/data/corpora/rcv1/test.npy"

def run(dtrain, dtest):
    lm = model.LinearModel(dtrain.dim, loss = bolt.ModifiedHuber(), reg = 0.00001)
    sgd = bolt.SGD(5)
    sgd.train(lm,dtrain,verbose=1)
    print "SGD fin. "

dtrain = io.MemoryDataset.load(ftrain, verbose = 0)
dtest = io.MemoryDataset.load(ftest, verbose = 0)
cProfile.runctx("run(dtrain, dtest)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
