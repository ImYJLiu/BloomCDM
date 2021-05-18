from constract_test.pmf import PMF
import os
import pickle
import random
import argparse
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score


class TripletUniformPair(IterableDataset): # 产生数据集 需要传入组信息
    def __init__(self, num_exer, pair, shuffle, num_epochs):
        self.num_exer = num_exer
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
        for k, v in self.pair[idx][1].items():
            i, s = int(k), int(v)
        return u, i, s


def aucc(user_emb, item_emb, train_user_list, test_user_list, batch=512):
    result = None
    for i in range(0, user_emb.shape[0], batch):  # 一批批的取出用户
        mask = user_emb.new_ones([min([batch, user_emb.shape[0] - i]), item_emb.shape[0]])  # 设置掩码
        for j in range(batch):
            if i + j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_user_list[i + j])).cuda().long(),
                             value=torch.tensor(0.0).cuda())
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i + min(batch, user_emb.shape[0] - i), :], item_emb.t())  # 用户和商品进行相乘
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)  # 【疑：为什么要把[训练集]已经发现的商品遮住呢？】
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)
    result = result.cpu()

    y_tests = [0 for i in range(user_emb.shape[0])]
    for i in range(user_emb.shape[0]):  # 每一个用户
        test = set(test_user_list[i])  # 属于用户的商品
        y_test = [0 for u in range(item_emb.shape[0])]
        for t in test:
            y_test[t] = 1
        y_tests[i] = y_test
    y_scores = result
    y_tests, y_scores = np.reshape(np.array(y_tests),newshape=(-1)), np.reshape(y_scores.numpy(),newshape=(-1))
    auc = roc_auc_score(y_tests, y_scores) # y_tests 真实值 y_scores 预测值
    rmse = np.sqrt(np.mean((y_tests - y_scores) ** 2))
    # # compute accuracy
    # accuracy = correct_count / exer_count
    print('auc:%6d' % auc)

def precision_and_recall_k(user_emb, item_emb, train_user_list, test_user_list, klist, batch=512):
    """Compute precision at k using GPU.

    Args:
        user_emb (torch.Tensor): embedding for user [user_num, dim]
        item_emb (torch.Tensor): embedding for item [item_num, dim]
        train_user_list (list(set)):
        test_user_list (list(set)):
        k (list(int)):
    Returns:
        (torch.Tensor, torch.Tensor) Precision and recall at k
    """
    # Calculate max k value
    max_k = max(klist)

    # Compute all pair of training and test record
    result = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_user_list[i+j])).cuda().long(), value=torch.tensor(0).cuda().long())
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        _, cur_result = torch.topk(cur_result, k=max_k, dim=1)
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)

    result = result.cpu()
    # Sort indice and get test_pred_topk
    precisions, recalls = [], []
    for k in klist:
        precision, recall = 0, 0
        for i in range(user_emb.shape[0]):
            test = set(test_user_list[i])
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            precision += val / max([min([k, len(test)]), 1])
            recall += val / max([len(test), 1])
        precisions.append(precision / user_emb.shape[0])
        recalls.append(recall / user_emb.shape[0])
    return precisions, recalls




def main(args):
    print('============================')
    print('Loading data')

    # Load preprocess data
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size, exer_size, know_size, higher_know_size = dataset['user_size'], dataset['exer_size'],dataset['know_size'],dataset['higher_know_size']
        train_dict_pair, know_group, higher_know_group\
            = dataset['train_dict_pair'],  dataset['know_group'], dataset['higher_know_group']
    print('Load complete')
    print('============================')


    # Create dataset, model, optimizer
    dataset = TripletUniformPair(exer_size, train_dict_pair, True, args.n_epochs)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True)
    model = PMF(exer_size, user_size,  args.batch_size, args.dim, args.lamdaU, args.lamdaV_1).cuda()
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
            print('idx: %d, loss: %.4f' % (idx, smooth_loss))
        idx += 1
    dirname = os.path.dirname(os.path.abspath(args.model))
    print('Save complete')
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), args.model)
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
                        default=20,
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
                        default=0.0001,
                        help="lamdaU factor")

    parser.add_argument('--lamdaV_1',
                        type=float,
                        default=0.01,
                        help="lamdaV_1 factor")

    parser.add_argument('--lr',
                        type=float,
                        default=5e-3,
                        help="Learning rate")

    parser.add_argument('--print_every',
                        type=int,
                        default=100,
                        help="Period for printing smoothing loss during training")

    parser.add_argument('--save_every',
                        type=int,
                        default=1000,
                        help="Period for saving model during training")

    parser.add_argument('--model',
                        type=str,
                        default="../data_edu/Assist/pmf/output_model.pt",
                        help="File path for model")
    args = parser.parse_args()
    main(args)