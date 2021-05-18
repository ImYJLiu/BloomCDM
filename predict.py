import os
import pickle
import argparse
import numpy as np
import torch
from BloomCDM import BloomCDM
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_model(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()

def aucc(user_emb, item_emb, train_right_list, test_right_list, batch=512):
    result = None
    for i in range(0, user_emb.shape[0], batch):  # 一批批的取出用户
        mask = user_emb.new_ones([min([batch, user_emb.shape[0] - i]), item_emb.shape[0]])  # 设置掩码
        for j in range(batch):
            if i + j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_right_list[i + j])).cuda().long(),
                             value=torch.tensor(0.0).cuda())
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i + min(batch, user_emb.shape[0] - i), :], item_emb.t())  # 用户和商品进行相乘
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)  # 【疑：为什么要把[训练集]已经发现的商品遮住呢？】
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)
    result = result.cpu()
    y_scores = result.float() #遮住训练集的预测结果

    y_tests = [0 for i in range(user_emb.shape[0])] # y_tests:真实 y_scores：预测
    for i in range(user_emb.shape[0]):  # 每一个学生
        test = set(test_right_list[i])   # 做对的题目
        y_test = [0 for u in range(item_emb.shape[0])]
        for t in test:
            y_test[t] = 1
        y_tests[i] = y_test


    correct_count = 0
    all_count = 0
    for i in range(y_scores.shape[0]):
        for j in range(y_scores.shape[1]):
            if (y_tests[i][j] == 1): # 如果是预测集中的数据
                all_count += 1
                if y_scores[i][j] - 0.5 > 0:
                    correct_count += 1


    y_tests, y_scores = np.reshape(np.array(y_tests), newshape=(-1)), np.reshape(y_scores.numpy(), newshape=(-1))

    # 根均方误差(RMSE)
    rmse = np.sqrt(mean_squared_error(y_tests, y_scores))
    mae = mean_absolute_error(y_tests, y_scores)
    acc = correct_count / all_count
    aucs = roc_auc_score(y_tests, y_scores) # y_tests 真实值 y_scores 预测值
    print('acc : %.6f' % acc, 'auc : %.6f' % aucs, '\n')
    print('rmse : %.6f' % rmse, 'mae : %.6f' % mae)


def main(args):
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size, exer_size, know_size, higher_know_size = dataset['user_size'], dataset['exer_size'], dataset['know_size'], dataset['higher_know_size']
        train_right_list, test_right_list = dataset['train_right_list'], dataset['test_right_list']
        train_pair, know_group, higher_know_group, all_sole_dict, all_higher_sole_dict = dataset['train_pair'], dataset['know_group'], dataset['higher_know_group'], dataset['all_sole_dict'], dataset['all_higher_sole_dict']
    print('Load complete')
    print('============================')

    model = BloomCDM(user_size, exer_size, know_size, higher_know_size, args.batch_size, know_group, higher_know_group,
                     args.dim, args.weight_decay, args.weight_decay_1, args.lamdaU, args.lamdaV_1).cuda()
    load_model(model, args.model)
    print('Load model complete')
    print("Start predict....")
    aucc(model.W.detach(), model.H.detach(), train_right_list, test_right_list, batch=512)
    print('Predict complete')


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
                        default=50,
                        help="Number of epoch during training")

    parser.add_argument('--batch_size',
                        type=int,
                        default=4096,
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
