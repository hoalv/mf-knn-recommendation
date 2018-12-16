from scipy.spatial import distance
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# a = (1, 2, 3)
# b = (1, 2,2)
# dst = distance.euclidean(a, b)
# print(dst)

# c =  np.array([1, 2, 3, 4])
# e= c.reshape(4,1)
# d =np.array([[1, 2, 3],[1,2,3],[1,2,3],[1,2,3]])
# tich = d*e
# print("tich: ", tich)

# kq = tich.sum(axis=0)
# print("kq: ", kq)

# print("e sum:", e.sum())

# d =np.array([[1, 2, 3],[1,2,5]])
# e =np.array([[1, 2, 3],[1,2,3]])
# print(cosine_similarity(d,e))

d =np.array([[1, 2, 3],[1,2,3],[1,2,3],[1,2,3]])
print(d[:, 0])