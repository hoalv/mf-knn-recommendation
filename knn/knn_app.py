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

in_file = open("MF.obj", "rb") # opening for [r]eading as [b]inary
mf = pickle.load(in_file) # if you only wanted to read 512 bytes, do .read(512)
in_file.close()
print("x:\n", mf.X.dot(mf.W))
print("xxx:\n",mf.Y_data_n)
# print("W: \n",mf.W.T[nearest_ids])

rs = KNN(users = users, n_users = n_users, users_train = users_train, user_new = user_new, k=5, mf = mf)

rs.fit()

rs.pred()

rate_train = rs.mf.Y_data_n[rs.mf.Y_data_n[:, 0]==1]
best_rate_ids = np.argsort(rate_train[:, 2])[-10:]
# print("rate_train:\n",  rate_train[best_rate_ids, 1])
# print("rate_train:\n",  rate_train)