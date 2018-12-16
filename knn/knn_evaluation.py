from knn import KNN
import sys
sys.path.append('../mf')
from mf import MF
import pickle
import numpy as np
import math
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../dataset/ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1').as_matrix()

n_users = users.shape[0]

user_occupations = pd.read_csv('../dataset/ml-100k/u.occupation', header=None, encoding='latin-1').as_matrix()

# replace field gender to number value
# all_users = users[:,1:4]
all_users = users[:, :4]
all_users[all_users[:,2]=='M', 2] = 1
all_users[all_users[:,2]=='F', 2] = 0

# replace field occupation to number value
n_occupations = user_occupations.shape[0]
for i in range(n_occupations):
	all_users[all_users[:,3]==user_occupations[i,0], 3] = i

print(all_users[:, 1:4])


in_file = open("MF.obj", "rb") # opening for [r]eading as [b]inary
mf = pickle.load(in_file) # if you only wanted to read 512 bytes, do .read(512)
in_file.close()
print("x:\n", mf.X.dot(mf.W).shape)
print("xxx:\n",mf.Y_data_n.shape)

users_train, users_test, f_train, f_test = train_test_split(all_users, mf.W.T, test_size=100)

# split tranning set to validation
users_train_v, users_validate, f_train_v, f_validate = train_test_split(users_train, f_train, test_size=100)

print("users_train:\n", users_train.shape)
print("W:\n", f_train.shape)

for j in range(5,100,20):
	rs = KNN(users = users, n_users = n_users, all_users = all_users[:, 1:4], users_train = users_train[:, 1:4], user_new = users_test[:, 1:4], k=j, mf = mf)

	rs.fit()


	for ii in range(users_test.shape[0]):
	# rs.pred(mf_k = 10, features_train = f_train, user_index =ii)
	# print("real f:\n", f_test[ii])
		rs.evaluate_RMSE(features_train=f_train, features_test=f_test, user_index=ii, user_id=users_test[ii][0])

	RMSE = np.sqrt(rs.SE/rs.total_rating_rmse)
	print("K , RMSE: ", j, RMSE)
