import numpy as np
import math as m
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class Cluster():
    def __init__(self, data_path, k):
        self.data_path = data_path
        self.k = k


    # 导入数据
    def load_data(self):
        points = np.loadtxt(self.data_path)
        return points


    def cal_dis(self, data, clu):
        """
        计算质点与数据点的距离
        :param data: 样本点
        :param clu:  质点集合
        :param k: 类别个数
        :return: 质心与样本点距离矩阵
        """
        dis = []
        for i in range(len(data)):
            dis.append([])
            for j in range(self.k):
                dis[i].append(np.around(m.sqrt((data[i, 0] - clu[j, 0])**2 + (data[i, 1]-clu[j, 1])**2), 2))
        return np.asarray(dis)


    def divide(self, data, dis):
        """
        对数据点分组
        :param data: 样本集合
        :param dis: 质心与所有样本的距离
        :param k: 类别个数
        :return: 分割后样本
        """
        clusterRes = [0] * len(data)
        for i in range(len(data)):
            seq = np.argsort(dis[i])
            clusterRes[i] = seq[0]

        return np.asarray(clusterRes)


    def center(self, data, clusterRes):
        """
        计算质心
        :param group: 分组后样本
        :param k: 类别个数
        :return: 计算得到的质心
        """
        clunew = []
        for i in range(self.k):
            # 计算每个组的新质心
            idx = np.where(clusterRes == i)
            sum = data[idx].sum(axis=0)
            avg_sum = sum/len(data[idx])
            clunew.append(avg_sum)
        clunew = np.asarray(clunew)
        return clunew[:, 0: 2]


    def classfy(self, data, clu):
        """
        迭代收敛更新质心
        :param data: 样本集合
        :param clu: 质心集合
        :param k: 类别个数
        :return: 误差， 新质心
        """
        clulist = self.cal_dis(data, clu)  # 计算质点与数据点的距离(通过余弦距离)
        clusterRes = self.divide(data, clulist) # 对数据点分组
        clunew = self.center(data, clusterRes) # 计算质心
        err = clunew - clu # 质心 - 质点
        return err, clunew, clusterRes

    def data_label(self, data, clusterRes):
        data_label = np.zeros((len(data), 3))
        nPoints = len(data)
        for i in range(self.k):
            for j in range(nPoints):
                if clusterRes[j] == i:
                    data_label[j,0] = data[j, 0]
                    data_label[j,1] = data[j, 1]
                    data_label[j,2] = i
        return data_label

    def plotRes(self, data, clusterRes):
        """
        结果可视化
        :param data:样本集
        :param clusterRes:聚类结果
        :param clusterNum: 类个数
        :return:
        """
        nPoints = len(data)
        scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
        for i in range(self.k):
            color = scatterColors[i % len(scatterColors)]
            x1 = [];  y1 = []
            for j in range(nPoints):
                if clusterRes[j] == i:
                    x1.append(data[j, 0])
                    y1.append(data[j, 1])
            plt.scatter(x1, y1, c=color, alpha=1, marker='+')
        plt.show()


    def getPCAData(self, data):
        pcaClf = PCA(self.k, whiten=True)
        pcaClf.fit(data)
        data_PCA = pcaClf.transform(data)  # 用来降低维度
        return data_PCA


    def modiData(self, data):
        x1 = []
        x2 = []
        for i in range(0, len(data + 1)):
            x1.append(data[i][0])
            x2.append(data[i][1])
        x1 = np.array(x1)
        x2 = np.array(x2)
        # 重塑数据
        X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
        return X


    # 绘制样式
    def drawKmodel(self, XData, t):
        plt.figure(figsize=(10, 10))
        colors = ['g', 'r', 'y', 'b']
        markers = ['o', 's', 'd', 'h']
        kmeans_model = KMeans(n_clusters=t).fit(XData)
        for i, l in enumerate(kmeans_model.labels_):
            plt.plot(XData[i][0], XData[i][1], color=colors[l], marker=markers[l], ls='None')
            plt.title('%s Countries K-Means' % (len(XData)))
            plt.savefig('kmeans_test.jpg')
            plt.show()



    # data = load_data()
    # data2017PCA = getPCAData(data, 2)
    # data2017X = modiData(data2017PCA)
    # data = data2017X
    # k = 50
    #
    # clu = random.sample(data[:, 0:2].tolist(), k)  # 随机取质心
    # clu = np.asarray(clu)
    # err, clunew,  k, clusterRes = classfy(data, clu, k) # 数据 + 质心
    # while np.any(abs(err) > 0):
    #     err, clunew,  k, clusterRes = classfy(data, clunew, k)
    #
    # clulist = cal_dis(data, clunew, k)
    # clusterResult = divide(data, clulist)
    # plotRes(data, clusterResult, k)