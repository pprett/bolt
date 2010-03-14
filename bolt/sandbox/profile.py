#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

import bolt

ftrain =  "/media/disk-3/data/corpora/rcv1/train.npy"
ftest = "/media/disk-3/data/corpora/rcv1/test.npy"

def runPegasos(dtrain, dtest):
    lm = bolt.LinearModel(dtrain.dim, biasterm = False)
    pegasos = bolt.PEGASOS(0.0001, 2)
    pegasos.train(lm,dtrain,verbose=1)
    print "Pegasos fin. "

def runSGD(dtrain, dtest):
    lm = bolt.LinearModel(dtrain.dim, biasterm = False)
    pegasos = bolt.SGD(bolt.ModifiedHuber(), 0.000001, 5)
    pegasos.train(lm,dtrain,verbose=1)
    print "SGD fin. "

dtrain = bolt.MemoryDataset.load(ftrain, verbose = 0)
dtest = bolt.MemoryDataset.load(ftest, verbose = 0)
cProfile.runctx("runPegasos(dtrain, dtest)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
