# import numpy as np
# import torch
#
# # user_list: 学生做过的题目
# # sole_list: 习题考察知识点
#
# class DOA():
#     def __init__(self, user_size, user_emb, item_emb, user_list):
#         self.user_size, self.user_emb, self.item_emb, self.user_list = user_size, user_emb, item_emb, user_list
#         self.stu2stu = torch.zeros((self.user_size, self.user_size))
#         self.knowledge_proficiency = torch.mm(self.user_emb, self.item_emb.t())
#
#
#     def doa(self,):
#         know = 0
#         j = 0
#         sum = 0
#         for i in range(self.user_size):
#             sum += self.delta(i,j)
#             for item in self.user_list[i]:
#                 if item == j:
#             j+=1
#
#
#
#     def delta(self, i,j, knowledge_proficiency):
#         if(knowledge_proficiency[i] > knowledge_proficiency[j]):
#             return 1
#         return 0
#
#     def J(self, i, j, k):
#
