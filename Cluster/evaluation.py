from sklearn.cluster import DBSCAN  # 进行DBSCAN聚类
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score ,calinski_harabasz_score,davies_bouldin_score # 计算 轮廓系数，CH 指标，DBI 
 

    
class E():
    def __init__(self,data, y_pred):
        self.data = data
        self.y_pred = y_pred
        self.DBI()
#         self.lunkuo()
#         self.CH()
        
    
#     def lunkuo(self,):
#         luokuoxishu = silhouette_score(self.data, self.y_pred, metric='euclidean')
#         print('luokuo:', luokuoxishu)
#         return luokuoxishu
    
    def CH(self,):
        CHxishu = calinski_harabasz_score(self.data, self.y_pred)
        print('CH:',CHxishu)
        return CHxishu
    
    def DBI(self,):
        DBIscore = davies_bouldin_score(self.data, self.y_pred) 
        print('DBI:',DBIscore)
        return DBIscore
