# -*- coding: utf-8 -*-
import os
import gzip
import json
import pickle
import random
from Cluster import Cluster
import argparse

import numpy as np
import pandas as pd

'''
 读取数据，转为pickle，并完成加载数据功能。
'''


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.dat')

    def load(self):
        # Load data
        df = pd.read_csv(self.fpath,
                         sep='::',
                         engine='python',
                         names=['user', 'item', 'rate', 'time'])
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        return df


class MovieLens20M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.csv')

    def load(self):
        df = pd.read_csv(self.fpath,
                         sep=',',
                         names=['user', 'item', 'rate', 'time'],
                         usecols=['user', 'item', 'time'],
                         skiprows=1)
        return df


class AmazonBeauty(DatasetLoader):
    def __init__(self, data_dir, file_name='All_Beauty.json.gz'):
        self.fpath = os.path.join(data_dir, file_name)

    def load(self):
        raw_list = []
        with gzip.open(self.fpath) as f:
            for idx, line in enumerate(f):
                raw_data = json.loads(line)
                raw_list.append({'user': raw_data['reviewerID'],
                                 'item': raw_data['asin'],
                                 'rate': raw_data['overall'],
                                 'time': raw_data['unixReviewTime']})
        df = pd.DataFrame(raw_list)
        print('Check if any column has null value')
        print(df.isnull().any())
        print('Total user number: %d' % df['user'].nunique())
        print('Total item number: %d' % df['item'].nunique())
        print('The number of unique item per user')
        print(df.groupby('user')['item'].nunique().value_counts())
        print('The number of unique user per item')
        print(df.groupby('item')['user'].nunique().value_counts())
        return df


class Assist(DatasetLoader):
    def __init__(self, data_dir, file_name='log_data.json'):
        self.fpath = os.path.join(data_dir, file_name)

    def load(self):
        print('================================================')
        print('Start load raw data')
        raw_list = []
        f_r = open(self.fpath, encoding='utf-8')
        f_r = json.loads(f_r.read())
        for l in f_r:
            for i in l['logs']:
                raw_list.append({'user': l['user_id'],
                                 'exer': i['exer_id'],
                                 'score': i['score'],
                                 'know': i['knowledge_code']})
        df = pd.DataFrame(raw_list)
        print('Check if any column has null value')
        print(df.isnull().any())
        print('Total user number: %d' % df['user'].nunique())
        print('Total item number: %d' % df['exer'].nunique())
        print('Loaded raw data')
        print('================================================')
        return df


class Gowalla(DatasetLoader):
    """Work In Progress"""

    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'loc-gowalla_totalCheckins.txt')

    def load(self):
        df = pd.read_csv(self.fpath,
                         sep='\t',
                         names=['user', 'time', 'latitude', 'longitude', 'item'],
                         usecols=['user', 'item', 'time'])
        df_size, df_nxt_size = 0, len(df)
        while df_size != df_nxt_size:
            # Update
            df_size = df_nxt_size
            # Remove user which doesn't contain at least five items to guarantee the existance of `test_item`
            groupby_user = df.groupby('user')['item'].nunique()
            valid_user = groupby_user.index[groupby_user >= 15].tolist()
            df = df[df['user'].isin(valid_user)]
            df = df.reset_index(drop=True)

            # Remove item which doesn't contain at least five users
            groupby_item = df.groupby('item')['user'].nunique()
            valid_item = groupby_item.index[groupby_item >= 15].tolist()
            df = df[df['item'].isin(valid_item)]
            df = df.reset_index(drop=True)

            # Update
            df_nxt_size = len(df)

        print('User distribution')
        print(df.groupby('user')['item'].nunique().describe())
        print('Item distribution')
        print(df.groupby('item')['user'].nunique().describe())
        return df


def convert_unique_idx(df, column_name):
    column_dict = {}
    if column_name == 'know':
        i = 0
        for id, knows in enumerate(df[column_name]):
            knows_id = []
            for know in knows:
                if know not in column_dict.keys():
                    column_dict[know] = i
                    knows_id.append(i)
                    i += 1
        df[column_name].loc[id] = knows_id
    else:
        column_dict = {x: i for i, x in enumerate(df[column_name].unique())}  # dict = {实际数值 ： 赋予的id}
        df[column_name] = df[column_name].apply(
            column_dict.get)  # TODO: 不太理解apply(columu_dict.get)是什么意思  将列值改为字典的value值
        df[column_name] = df[column_name].astype('int')
        assert df[column_name].min() == 0
        assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def gen_sole_know(df):
    def get_sole(v):
        if len(v) != 0:
            return v[0]
        else:
            return np.NAN

    df['sole_know'] = df['know'].apply(get_sole)
    return df


def cosine_similarity(x, y, norm=False):
    dot_product, square_sum_x, square_sum_y = 0, 0, 0
    for i in range(len(x)):
        dot_product += x[i] * y[i]
        square_sum_x += x[i] * x[i]
        square_sum_y += y[i] * y[i]
    cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def create_group_1(df):
    know_group = {}  # 字典嵌套
    for i, row in df.iterrows():
        user = row['user']  # user
        sole_know = row['sole_know']  # sole_know
        score = row['score']  # score
        if user not in know_group:
            user_dict = {}
            msg = {}
            msg['sum'] = score
            msg['num'] = 1
            user_dict[sole_know] = msg
            know_group[user] = user_dict
        elif sole_know not in know_group[user]:
            msg = {}
            msg['sum'] = score
            msg['num'] = 1
            know_group[user][sole_know] = msg
        else:
            know_group[user][sole_know]['num'] += 1
            know_group[user][sole_know]['sum'] += score
    print('Complete create know group')
    for user, user_dict in know_group.items():
        for sole_know, msg in user_dict.items():
            msg['avg_sum'] = np.round(msg['sum'] / msg['num'], 2)
        know_group[user][sole_know] = msg
    return know_group


def getCluster(data_path, k):
    cluster = Cluster(data_path, k)
    data = cluster.load_data()
    data2017PCA = cluster.getPCAData(data)
    data2017X = cluster.modiData(data2017PCA)
    data = data2017X
    clu = random.sample(data[:, 0:2].tolist(), cluster.k)  # 随机取质心
    clu = np.asarray(clu)
    err, clunew, clusterRes = cluster.classfy(data, clu)  # 数据 + 质心
    while np.any(abs(err) > 0):
        err, clunew, clusterRes = cluster.classfy(data, clunew)

    clulist = cluster.cal_dis(data, clunew)
    clusterResult = cluster.divide(data, clulist)
    data_label = cluster.data_label(data, clusterResult)
    return data_label


def create_group_2(user_size, know_size, know_group):
    know_user = np.around(np.zeros((know_size, user_size)), 3)
    for user, user_dict in know_group.items():
        for sole_know, msg in user_dict.items():
            know_user[sole_know, user] = msg['avg_sum']

    np.savetxt('know_user_kmeans.txt', know_user, fmt='%.03f')
    higher_know_size = 50
    data_label = getCluster('know_user_kmeans.txt', higher_know_size)
    '''
    higher_know_group：
        用户：
            高阶知识组：
                'sum':分数
                'count':个数
                'avg_sum':平均分数
    '''
    all_higher_sole_dict = {}
    higher_know_group = {}
    for user, user_dict in know_group.items():
        all_higher_sole_dict[user] = []
        for sole_know, msg in user_dict.items():
            sole_higher_know = int(data_label[sole_know][2])
            all_higher_sole_dict[user].append(sole_higher_know)
            if user not in higher_know_group:
                user_higher_dict = {}
                higher_msg = {}
                higher_msg['sum'] = msg['avg_sum']
                higher_msg['num'] = 1
                user_higher_dict[sole_higher_know] = higher_msg
                higher_know_group[user] = user_higher_dict
            elif sole_higher_know not in higher_know_group[user]:
                higher_msg = {}
                higher_msg['sum'] = msg['avg_sum']
                higher_msg['num'] = 1
                higher_know_group[user][sole_higher_know] = higher_msg
            else:
                higher_know_group[user][sole_higher_know]['num'] += 1
                higher_know_group[user][sole_higher_know]['sum'] += msg['avg_sum']
    for user, user_higher_dict in higher_know_group.items():
        for sole_higher_know, higher_msg in user_higher_dict.items():
            higher_msg['avg_sum'] = np.around(higher_msg['sum'] / higher_msg['num'], 2)
        higher_know_group[user][sole_higher_know] = higher_msg
    print('Complete create know group 2')
    return higher_know_group, higher_know_size, all_higher_sole_dict


def create_user_list(df, user_size, exer_size):
    user_list = [list() for u in range(user_size)]
    user_right_list = [list() for u in range(user_size)]
    user_dict_list = [list() for u in range(user_size)]
    user_item_score = np.zeros((user_size, exer_size))
    for row in df.itertuples():
        user_list[row.user].append(row.exer)
        user_dict_list[row.user].append({row.exer: row.score})
        user_item_score[row.user][row.exer] = row.score
        if row.score == 1:
            user_right_list[row.user].append(row.exer)
    return user_list, user_right_list, user_dict_list, user_item_score


def split_train_test(df, user_size, exer_size, test_size=0.2, time_order=False):
    """Split a dataset into `train_user_list` and `test_user_list`.
    Because it needs `user_list` for splitting dataset as `time_order` is set,
    Returning `user_list` data structure will be a good choice."""
    # TODO: Handle duplicated items

    test_idx = np.random.choice(len(df), size=int(len(df) * test_size))
    train_idx = list(set(range(len(df))) - set(test_idx))
    test_df = df.loc[test_idx].reset_index(drop=True)
    train_df = df.loc[train_idx].reset_index(drop=True)
    test_user_list, test_right_list, test_user_dict_list, test_user_item_score = create_user_list(test_df, user_size, exer_size)
    train_user_list, train_right_list, train_user_dict_list, train_user_item_score = create_user_list(train_df, user_size, exer_size)

    return train_user_list, train_right_list, train_user_dict_list, train_user_item_score, train_df, test_user_list, test_right_list, test_user_dict_list, test_user_item_score, test_df


def create_pair(user_dict_list):
    pair = []
    for user, item_list in enumerate(user_dict_list):
        pair.extend([(user, item) for item in item_list])
    return pair


def main(args):
    if args.data_name == 'ml-1m':
        df = MovieLens1M(args.data_dir).load()
    elif args.data_name == 'ml-20m':
        df = MovieLens20M(args.data_dir).load()
    elif args.data_name == 'amazon-beauty':
        df = AmazonBeauty(args.data_dir).load()
    elif args.data_name == 'assist':
        df = Assist(args.tmp_dir).load()
    else:
        raise NotImplementedError
    # 转为唯一的index
    df, user_mapping = convert_unique_idx(df, 'user')
    df, exer_mapping = convert_unique_idx(df, 'exer')  # 保存转换id之后的dataframe
    df = gen_sole_know(df)
    df, know_mapping = convert_unique_idx(df, 'sole_know')  # 对sole_know进行id的重标
    df.dropna(inplace=True)
    print('Complete assigning unique index to user, exer, sole_know')
    np.set_printoptions(precision=3)
    neg_pos = list(df.groupby(df['score']))
    print('all', len(df['score']))
    print('neg', len(neg_pos[0]))
    print('pos', len(neg_pos[1]))



if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()

    # 处理数据参数
    parser.add_argument("-dp", "--do_preprocess", type=bool,
                        help="True or False to preprocess raw data for bloomCDM (default = True)",
                        default=True)  # 是否处理数据
    parser.add_argument("-dn", "--data_name", type=str, default="assist",
                        help="Name of data.")
    parser.add_argument("-od", "--output_data", type=str, default="data_edu/Assist/data.pkl",
                        help="Output of data.")
    parser.add_argument("-td", "--tmp_dir", type=str, default="./data_edu/Assist/",
                        help="Dirction of data.")
    parser.add_argument("-ulc", "--user_limit_condition", nargs='+', type=int,
                        help="Limit condition of user.")
    parser.add_argument("-ulcf", "--user_limit_condition_f1", type=float,
                        help="Limit condition of user.")
    parser.add_argument("-plc", "--problem_limit_condition", nargs='+', type=int,
                        help="Limit condition of problem.")
    parser.add_argument("-plcf", "--problem_limit_condition_f1", type=float,
                        help="Limit condition of problem.")
    parser.add_argument("-tlc", "--time_limit_condition", type=str, action="append",
                        help="Limit condition of time.")
    parser.add_argument("-tdd", "--timedivided", type=int,
                        help="The number of timedivided.")

    args = parser.parse_args()
    do_preprocess = args.do_preprocess
    DataName = args.data_name
    Output_data = args.output_data

    # 输出设置的信息
    print('--------------------------------------------------------------------------------')
    print("-------------------Preprocess Data Option Setting-------------------")
    print("True or False to preprocess raw data for bloomCDM:", do_preprocess)
    print("Name of data(DataName): ", DataName)
    print("Output of data(Output_data): ", Output_data)
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")

    args = parser.parse_args()

    main(args)
