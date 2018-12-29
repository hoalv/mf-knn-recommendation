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
from sklearn.model_selection import KFold
import os
import time

start_time = time.time()

MOVIELENS_DIR = '../dataset/ml-1m'
USER_DATA_FILE = 'users.dat'
users_data = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
# users['age_desc'] = users['age'].apply(lambda x: AGES[x])
# users['occ_desc'] = users['occupation'].apply(lambda x: OCCUPATIONS[x])
# print (len(users), 'descriptions of', max_userid, 'users loaded.')
# 
# 
users = users_data.as_matrix()
n_users = users.shape[0]

all_users = users[:, :4]
all_users[all_users[:,1]=='M', 1] = 1
all_users[all_users[:,1]=='F', 1] = 0
print(all_users[:, 1:4])
print(n_users)


in_file = open("MF_1m.obj", "rb") # opening for [r]eading as [b]inary
mf = pickle.load(in_file) # if you only wanted to read 512 bytes, do .read(512)
in_file.close()
print("x:\n", mf.X.dot(mf.W).shape)
print("xxx:\n",mf.Y_data_n.shape)

users_train, users_test, f_train, f_test = train_test_split(all_users, mf.W.T, test_size=0.9)

# split tranning set to validation
users_train_v, users_validate, f_train_v, f_validate = train_test_split(users_train, f_train, test_size=0.3)

print("users_train:\n", users_train.shape)
print("W:\n", f_train.shape)

rs = KNN(users = users, n_users = n_users, all_users = all_users[:, 1:4], users_train = users_train[:, :4], user_new = users_test[:, 1:4], k=50, mf = mf)

rs.fit()

print("--- %s seconds ---" % (time.time() - start_time))

for ii in range(users_test.shape[0]):
	# rs.pred(mf_k = 10, features_train = f_train, user_index =ii)
	# print("real f:\n", f_test[ii])
	rs.evaluate_RMSE(features_train=f_train, features_test=f_test, user_index=ii, user_id=users_test[ii][0])
	rs.item_average_evaluate_RMSE(features_train=f_train, features_test=f_test, user_index=ii, user_id=users_test[ii][0])
	rs.global_average_evaluate_RMSE(users_train = users_train[:, :4], user_index=ii, user_id=users_test[ii][0])

RMSE = np.sqrt(rs.SE/rs.total_rating_rmse)
print("RMSE: ", RMSE)

RMSE_avg = np.sqrt(rs.SE_average/rs.total_rating_rmse_average)
print("RMSE_avg: ", RMSE_avg)

RMSE_global = np.sqrt(rs.SE_global/rs.total_rating_rmse_global)
print("RMSE_global: ", RMSE_global)