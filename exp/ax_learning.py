import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from data_provider_.data_factory_ import data_provider
from torch import optim
import os
import time

class AuxiliaryLearningModel(object):
    def __init__(self, args):
        self.args = args

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[self.args.freq]
        
        self.m1 = nn.Linear(d_inp, 512)
        self.m2 = nn.Linear(512, 512)
        self.m3 = nn.Linear(512, d_inp)
    

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            device = torch.device('cuda:0')
            iter_count = 0
            train_loss = []

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            epoch_time = time.time()
            for i, (
            batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_mark_sequentail, seq_y_mark_sequentail) in enumerate(
                    train_loader):
                iter_count += 1
                model_optim.zero_grad()

                x = seq_x_mark_sequentail.float().to(device)
                y = seq_y_mark_sequentail.float().to(device)

                prediction = self.m1(x) 
                prediction = self.m2(prediction)
                prediction = self.m3(prediction)
            
                cost = F.mse_loss(prediction, y)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()


        torch.save(self.m2.state_dict(), './layer.pt')
        return