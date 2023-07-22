import torch
import argparse
import os

from exp.ax_learning import AuxiliaryLearningModel
import random
import numpy as np
from utils.tools import dotdict
import pandas as pd

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
print(torch.cuda.current_device())

torch.cuda.is_available()
torch.cuda.set_device(0)


# args = parser.parse_args()
args = dotdict()
args.target = 'OT'  
args.des = 'test'
#args.dropout = 0.05
args.num_workers = 0
args.gpu = 0
args.lradj = 'type1'
args.devices = '0'
args.use_gpu = True
args.use_multi_gpu = False
args.freq = 'h'
args.root_path = './data/ETT/'
args.data_path ='ETTh1.csv'
args.model_id='ETTh1_192_18'
args.data = 'ETTh1'
args.features = 'M'
args.len = 10 
args.time_point = 1 
args.ax_l = True
args.seq_len = 192 
args.label_len = 96 
args.pred_len = 18
args.itr = 1
args.learning_rate = 0.001
args.batch_size = 32
args.embed = 'timeF'
args.loss = 'mse'
args.train_epochs = 10
print('Args in experiment:')
print(args)

Exp = AuxiliaryLearningModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.freq,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.len,
        args.time_point,
        args.ax_l,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print(1)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
