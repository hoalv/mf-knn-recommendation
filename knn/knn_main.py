from knn import KNN
import sys
sys.path.append('../mf')
from mf import MF
import pickle
import numpy as np
import math
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../dataset/ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1').as_matrix()

n_users = users.shape[0]

user_occupations = pd.read_csv('../dataset/ml-100k/u.occupation', header=None, encoding='latin-1').as_matrix()

# replace field gender to number value
users_train = users[:,1:4]
users_train[users_train[:,1]=='M', 1] = 1
users_train[users_train[:,1]=='F', 1] = 0

# replace field occupation to number value
n_occupations = user_occupations.shape[0]
for i in range(n_occupations):
	users_train[users_train[:,2]==user_occupations[i,0], 2] = i

print(users_train)

user_new = np.array([53, 0, 13]).reshape(1, users_train.shape[1])
print("new:\n", user_new)
user_new = (user_new - np.min(users_train, axis=0))/np.ptp(users_train, axis = 0)
print("user_new: \n", user_new)

# nomalize user_train
users_train = (users_train - np.min(users_train, axis=0))/np.ptp(users_train, axis = 0)
# print(users_train)

# similarity matrix
S = cosine_similarity(user_new, users_train)
# print("sim: \n", S)

# find the k most similarity users
n=1
k=5
mf_k=10
a = np.argsort(S[0,:])[-k:] 
# and the corresponding similarity levels
nearest_s = S[0, a]
print(nearest_s)

#calculate feature of user_n
in_file = open("MF.obj", "rb") # opening for [r]eading as [b]inary
rs = pickle.load(in_file) # if you only wanted to read 512 bytes, do .read(512)
in_file.close()
print("x:\n", rs.X)
print("W: \n",rs.W.T[a])
nearest_s = np.reshape(nearest_s, (1,k))
print("nearest_s:\n",nearest_s)
print(nearest_s)
print(rs.W.T[a].T)


tich = rs.W.T[a].T*nearest_s
print("tich: \n", tich)


feature_user_n = (tich.sum(axis=1))/(np.abs(nearest_s).sum() + 1e-8)
print("feature: \n", feature_user_n)
feature_user_n = np.reshape(feature_user_n, (mf_k,1))
print("rating:\n",rs.X.dot(feature_user_n))

# A = np.array([23, 1, 20]).reshape(1,3)
# print(A)
# A_train = np.array([[23,1,20],[22,0,19],[23,1,20]]).reshape(3,3)
# sim= cosine_similarity(A, A_train)
# print(sim)
# k_ne = np.argsort(sim[0])[-2:]
# print(sim[0,k_ne])