import numpy as np
import math
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

class KNN(object):
	"""docst
	ring for KNN"""
	def __init__(self, users, n_users, all_users, users_train, user_new, k, mf = None, dist_func = cosine_similarity):
		self.users = users
		self.n_users = n_users
		self.all_users = all_users
		self.users_train = users_train[:, 1:4]	
		self.user_new = user_new
		self.users_train_raw = users_train
		# reshape(1, self.all_users.shape[1])
		self.k = k
		self.mf = mf
		self.dist_func = dist_func
		self.SE = 0

		# self.ratings_avg = np.mean(self.mf.Y_data_n[:, 2])
		self.total_rating_rmse = 0
		self.SE_average = 0
		self.total_rating_rmse_average = 0
		self.SE_global = 0
		self.total_rating_rmse_global = 0

		users_train_list = users_train[:, 0].tolist()
		user_ids = np.where(self.mf.Y_data_n[:, 0] == i-1 for i in users_train_list)[0]
		self.ratings_avg =  np.mean(self.mf.Y_data_n[user_ids, 2])
	
	def nomarlize_users(self):
		self.user_new = (self.user_new - np.min(self.all_users, axis=0))/np.ptp(self.all_users, axis = 0)
		self.users_train = (self.users_train - np.min(self.all_users, axis=0))/np.ptp(self.all_users, axis = 0)

	def similarity(self):
		self.S = self.dist_func(self.user_new, self.users_train)
		# print("S:\n", self.S)

	def refresh(self):
		"""
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
		self.nomarlize_users()
		self.similarity()

	def fit(self):
		self.refresh()

	def __caculate_feature_user_new(self, features_train, user_index):
        # find the k most similarity users
		nearest_ids = np.argsort(self.S[user_index])[-self.k:] 
		# and the corresponding similarity levels
		nearest_s = self.S[user_index, nearest_ids]
        # How did each of 'near' users rated item i
		nearest_s = np.reshape(nearest_s, (self.k, 1))
		# print("nearest_s: ", nearest_s)
		# print("features_train[nearest_ids]:\n", features_train[nearest_ids])
		# tich = self.mf.W.T[nearest_ids].T*nearest_s
		tich = features_train[nearest_ids]*nearest_s
		# print("tich: \n", tich)
		feature_user_new = (tich.sum(axis=0))/(np.abs(nearest_s).sum())
		return feature_user_new
    
	def pred(self, mf_k , features_train, user_index, user_id):
		# print("self.users_train_raw[0]: ", self.users_train_raw[:, 0])
		# print("user_id: ", user_id)
		if (user_id in self.users_train_raw[: , 0]):
			print("old user")
			id_in_users_train = np.where(self.users_train_raw[:, 0] == user_id)
			u_w = self.mf.W.T[id_in_users_train]
			return self.mf.X.dot(u_w.T)
		else:
			print("new user")
			feature_user_new = self.__caculate_feature_user_new(features_train, user_index)
			return self.mf.X.dot(feature_user_new.T)
		# feature_user_new = np.reshape(feature_user_new, (mf_k,1))
		# rate_pred = self.mf.X.dot(feature_user_new) 
		# print("rating:\n",rate_pred)
		# best_rate_ids = np.argsort(rate_pred[:, 0])[-10:]
		# print("best rate:\n", best_rate_ids )

	# Đánh giá kết quả bằng cách đo Root Mean Square Error:
	def evaluate_RMSE(self, features_train, features_test, user_index, user_id):
		feature_user_new = self.__caculate_feature_user_new(features_train, user_index)
		pred_rating = self.pred(mf_k = 100, features_train = features_train, user_index =user_index, user_id = user_id)
		# feature_user_new = np.reshape(feature_user_new, (mf_k,1))
		n_tests = features_test.shape[0]
    	 # squared error
		# self.SE += distance.euclidean(feature_user_new, features_test[user_index])**2
		ids = np.where(self.mf.Y_data_n[:, 0] == user_id-1)[0]
		real_rating_of_u = self.mf.Y_data_n[ids, 2]
		item_ids = self.mf.Y_data_n[ids, 1]
		for i in range(real_rating_of_u.shape[0]):
			# print("real_rating_of_u[i]:]\n", real_rating_of_u[i])
			# print("pred_rating[item_ids[i]-1]\n", pred_rating[item_ids[i]-1])
			if np.isnan(pred_rating[item_ids[i]-1]):
				pred_rating[item_ids[i]-1] = 0  
			self.SE += distance.euclidean(real_rating_of_u[i], pred_rating[item_ids[i]-1])**2
			self.total_rating_rmse += 1

	# Đánh giá kết quả bằng cách đo Root Mean Square Error:
	def item_average_evaluate_RMSE(self, features_train, features_test, user_index, user_id):
		# feature_user_new = self.__caculate_feature_user_new(features_train, user_index)
		# pred_rating = self.pred(mf_k = 100, features_train = features_train, user_index =user_index)
		# # feature_user_new = np.reshape(feature_user_new, (mf_k,1))
		# n_tests = features_test.shape[0]
    	 # squared error
		# self.SE += distance.euclidean(feature_user_new, features_test[user_index])**2
		ids = np.where(self.mf.Y_data_n[:, 0] == user_id-1)[0]
		real_rating_of_u = self.mf.Y_data_n[ids, 2]
		item_ids = self.mf.Y_data_n[ids, 1]
		for i in range(real_rating_of_u.shape[0]):
			# print("real_rating_of_u[i]:]\n", real_rating_of_u[i])
			# print("pred_rating[item_ids[i]-1]\n", pred_rating[item_ids[i]-1])
			self.SE_average += distance.euclidean(real_rating_of_u[i], self.get_average_rating_of_item(item_id = item_ids[i]))**2
			self.total_rating_rmse_average += 1

	# Đánh giá kết quả bằng cách đo Root Mean Square Error:
	def global_average_evaluate_RMSE(self, users_train, user_index, user_id):
		# feature_user_new = self.__caculate_feature_user_new(features_train, user_index)
		# pred_rating = self.pred(mf_k = 100, features_train = features_train, user_index =user_index)
		# # feature_user_new = np.reshape(feature_user_new, (mf_k,1))
		# n_tests = features_test.shape[0]
    	 # squared error
		# self.SE += distance.euclidean(feature_user_new, features_test[user_index])**2
		
		users_train_list = users_train[:, 0].tolist()
		user_ids = np.where(self.mf.Y_data_n[:, 0] == i-1 for i in users_train_list)[0]
		self.ratings_avg =  np.mean(self.mf.Y_data_n[user_ids, 2])
		ids = np.where(self.mf.Y_data_n[:, 0] == user_id)[0]
		real_rating_of_u = self.mf.Y_data_n[ids, 2]
		for i in range(real_rating_of_u.shape[0]):
			# print("real_rating_of_u[i]:]\n", real_rating_of_u[i])
			# print("pred_rating[item_ids[i]-1]\n", pred_rating[item_ids[i]-1])
			self.SE_global += distance.euclidean(real_rating_of_u[i], self.ratings_avg)**2
			self.total_rating_rmse_global += 1

	def get_average_rating_of_item(self, item_id):
		ids = np.where(self.mf.Y_data_n[:, 1] == item_id)[0]
		rating_of_item = self.mf.Y_data_n[ids, 2]
		m = np.mean(rating_of_item)
		if np.isnan(m):
			m=0
		return m

	def get_average_rating_global(self):
		ratings = self.mf.Y_data_n[:, 2]
		m = np.mean(ratings)
		return m  