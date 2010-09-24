#!/usr/bin/python2.6
#
# Copyright (C) 2010 Peter Prettenhofer.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""
Bolt
====

Bolt online learning toolbox.

Documentation is available in the docstrings. 

Subpackages
-----------

model
   Model specifications. 

trainer
   Extension module containing various model trainers.

io
   Input/Output routines; reading datasets, writing predictions.

eval
   Evaluation metrics. 

parse
   Command line parsing.

see http://github.com/pprett/bolt

"""

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"
__copyright__ = "Apache License v2.0"

import eval

from trainer import OVA
from trainer.sgd import predict,SGD,LossFunction,Classification,Regression,Hinge, ModifiedHuber, Log, SquaredError, Huber, PEGASOS
from trainer.maxent import MaxentSGD
from trainer.avgperceptron import AveragedPerceptron
from io import MemoryDataset, sparsedtype, dense2sparse, fromlist
from model import LinearModel, GeneralizedLinearModel
from eval import errorrate

__version__ = "1.4"

