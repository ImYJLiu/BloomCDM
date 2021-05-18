from constract_test.irt import IRT
import os
import pickle
import random
import argparse
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc


class TripletUniformPair(IterableDataset): # 产生数据集 需要传入组信息
    def __init__(self, num_exer,  user_list, pair, shuffle, num_epochs):
        self.num_exer = num_exer
        self.user_list = user_list
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs

    def __iter__(self):
        worker_info = get_worker_info()
        # Shuffle per epoch
        self.example_size = self.num_epochs * len(self.pair)
        self.example_index_queue = deque([])
        self.seed = 0
        if worker_info is not None:
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.example_size:
            raise StopIteration
        while len(self.example_index_queue) == 0:
            index_list = list(range(len(self.pair))) #所有pair的长度
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            if self.start_list_index is not None:
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.example_index_queue.extend(index_list)
        result = self._example(self.example_index_queue.popleft())
        self.index += self.num_workers
        return result

    def _example(self, idx):
        u = self.pair[idx][0]
        for k, v in self.pair[idx][1].items():
            i, s = int(k), int(v)
        # i = self.pair[idx][1]
        # s = self.pair[idx][2]
        return u, i, s



def main(args):
    print('============================')
    print('Loading data')

    # Load preprocess data
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size, exer_size, know_size, higher_know_size = dataset['user_size'], dataset['exer_size'],dataset['know_size'],dataset['higher_know_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        train_dict_pair, know_group, higher_know_group = \
            dataset['train_dict_pair'],  dataset['know_group'], dataset['higher_know_group']
    print('Load complete')
    print('============================')

    # Create dataset, model, optimizer
    dataset = TripletUniformPair(exer_size, train_user_list, train_dict_pair, True, args.n_epochs)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True)
    model = IRT(exer_size, user_size, args.batch_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    # Training
    smooth_loss = 0
    idx = 0
    print('============================')
    print('Start train')
    for u, i, s in loader:
        optimizer.zero_grad()
        loss = model(u, i, s)
        loss.backward()
        optimizer.step()
        writer.add_scalar('./train/loss', loss, idx)
        smooth_loss = smooth_loss * 0.99 + loss * 0.01
        if idx % args.print_every == (args.print_every - 1):
            print('loss: %.4f' % smooth_loss)
        idx += 1
    dirname = os.path.dirname(os.path.abspath(args.model))
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), args.model)
    print('Save model complete')
    print('Train complete')
    print('============================')


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default="../data_edu/Assist/data.pkl",
                        help="File path for data")

    parser.add_argument('--n_epochs',
                        type=int,
                        default=5,
                        help="Number of epoch during training")

    parser.add_argument('--batch_size',
                        type=int,
                        default=200,
                        help="Batch size in one iteration")

    parser.add_argument('--dim',
                        type=int,
                        default=4,
                        help="Dimension for embedding")

    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.025,
                        help="Weight decay factor")

    parser.add_argument('--weight_decay_1',
                        type=float,
                        default=0.025,
                        help="Weight decay_1 factor")

    parser.add_argument('--lamdaU',
                        type=float,
                        default=0.01,
                        help="lamdaU factor")

    parser.add_argument('--lamdaV_1',
                        type=float,
                        default=0.01,
                        help="lamdaV_1 factor")

    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help="Learning rate")

    parser.add_argument('--print_every',
                        type=int,
                        default=100,
                        help="Period for printing smoothing loss during training")

    parser.add_argument('--model',
                        type=str,
                        default="../data_edu/Assist/irt/output_model.pt",
                        help="File path for model")
    args = parser.parse_args()
    main(args)