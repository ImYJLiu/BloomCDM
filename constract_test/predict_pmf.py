import os
import pickle
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from constract_test.pmf import PMF



def load_model(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def aucc(user_emb, item_emb, train_user_list, test_user_list, test_user_item_score, batch):
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

    y_tests = [0 for i in range(user_emb.shape[0])] # y_tests:真实 y_scores：预测
    for i in range(user_emb.shape[0]):  # 每一个学生
        test = set(test_user_list[i])   # 做过的题目
        y_test = [0 for u in range(item_emb.shape[0])]
        for t in test:
            y_test[t] = 1
        y_tests[i] = y_test
    y_scores = result


    correct_count = 0
    all_count = 0
    y_tests_only = []
    y_scores_only = []
    for i in range(y_scores.shape[0]):
        for j in range(y_scores.shape[1]):
            if (y_tests[i][j] == 1):
                all_count += 1
                y_tests_only.append(test_user_item_score[i][j])
                y_scores_only.append(y_scores[i][j])
                if (test_user_item_score[i][j] == 1 and y_scores[i][j] > 0.5) or (test_user_item_score[i][j] == 0 and y_scores[i][j] < 0.5):
                    correct_count += 1

    # 根均方误差(RMSE)
    rmse = np.sqrt(mean_squared_error(y_tests_only, y_scores_only))
    mae = mean_absolute_error(y_tests_only, y_scores_only)
    acc = correct_count / all_count
    aucs = roc_auc_score(y_tests_only, y_scores_only) # y_tests 真实值 y_scores 预测值
    print(y_tests_only[:100])
    print(y_scores_only[:100])
    print('acc : %.6f' % acc, 'auc : %.6f' % aucs)
    print('rmse : %.6f' % rmse, 'mae : %.6f' % mae)




def precision_and_recall_k(user_emb, item_emb, train_user_list, test_user_list, klist, batch):
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
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_user_list[i+j])).cuda().long(), value=torch.tensor(0.0).cuda().long())
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

    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size, exer_size, know_size, higher_know_size = dataset['user_size'], dataset['exer_size'],dataset['know_size'],dataset['higher_know_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        train_user_item_score, test_user_item_score = dataset['train_user_item_score'], dataset['test_user_item_score']
    print('Load complete')

    model = PMF(exer_size, user_size,  args.batch_size, args.dim, args.lamdaU, args.lamdaV_1).cuda()

    load_model(model, args.model)
    print('Load model complete')
    print('============================')
    print("Start predict....")
    aucc(model.W.detach(), model.H.detach(), train_user_list, test_user_list, test_user_item_score,512)
    print('Predict complete')


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default="../data_edu/Assist/data.pkl",
                        help="File path for data")

    parser.add_argument('--n_epochs',
                        type=int,
                        default=7,
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