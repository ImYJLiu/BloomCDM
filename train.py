import os
import random
import pickle
import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from BloomCDM import BloomCDM
from torch.utils.tensorboard import SummaryWriter


class TripletUniformPair(IterableDataset):
    def __init__(self, num_exer, all_sole_dict, all_higher_sole_dict, user_right_list, pair, shuffle, num_epochs):
        self.num_exer = num_exer
        self.all_sole_dict = all_sole_dict
        self.all_higher_sole_dict = all_higher_sole_dict
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.user_right_list = user_right_list

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
            index_list = list(range(len(self.pair)))  # 所有pair的长度
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            if self.start_list_index is not None:
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (
                            self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.example_index_queue.extend(index_list)
        result = self._example(self.example_index_queue.popleft())
        self.index += self.num_workers
        return result

    def _example(self, idx):
        u = self.pair[idx][0]
        i = self.pair[idx][1]
        j = np.random.randint(self.num_exer)
        i_1 = random.sample(self.all_sole_dict[u].tolist(), 1)[0]
        i_2 = random.sample(self.all_higher_sole_dict[u], 1)[0]
        while j in self.user_right_list[u]:
            j = np.random.randint(self.num_exer)
        return u, i, j, i_1, i_2


def main(args):
    # Initialize seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size, exer_size, know_size, higher_know_size = dataset['user_size'], dataset['exer_size'], dataset['know_size'], dataset['higher_know_size']
        train_right_list, test_right_list = dataset['train_right_list'], dataset['test_right_list']
        train_pair, know_group, higher_know_group, all_sole_dict, all_higher_sole_dict = dataset['train_pair'], dataset['know_group'], dataset['higher_know_group'], dataset['all_sole_dict'], dataset['all_higher_sole_dict']
    print('Load complete')
    print('============================')

    # Create dataset, model, optimizer
    dataset = TripletUniformPair(exer_size, all_sole_dict, all_higher_sole_dict, train_right_list, train_pair, True, args.n_epochs)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)
    model = BloomCDM(user_size, exer_size, know_size, higher_know_size, args.batch_size, know_group, higher_know_group, args.dim, args.weight_decay, args.weight_decay_1, args.lamdaU, args.lamdaV_1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    # Training
    smooth_loss = 0
    idx = 0
    for u, i, j, i_1, i_2 in loader:
        optimizer.zero_grad()
        loss = model(u, i, j, i_1, i_2, args.batch_size)
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/loss', loss, idx)
        smooth_loss = smooth_loss*0.99 + loss*0.01
        if idx % args.print_every == (args.print_every - 1):
            print('loss: %.4f' % smooth_loss)
        idx += 1
    dirname = os.path.dirname(os.path.abspath(args.model))
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), args.model)
    print('Train compelete')

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default="data_edu/Assist/data.pkl",
                        help="File path for data")
    # Seed
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Seed (For reproducability)")
    # Model
    parser.add_argument('--dim',
                        type=int,
                        default=4,
                        help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help="Learning rate")

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
    # Training
    parser.add_argument('--n_epochs',
                        type=int,
                        default=60,
                        help="Number of epoch during training")

    parser.add_argument('--batch_size',
                        type=int,
                        default=200,
                        help="Batch size in one iteration")

    parser.add_argument('--print_every',
                        type=int,
                        default=20,
                        help="Period for printing smoothing loss during training")

    parser.add_argument('--model',
                        type=str,
                        default='./output_model.pt',
                        help="File path for model")
    args = parser.parse_args()
    main(args)
