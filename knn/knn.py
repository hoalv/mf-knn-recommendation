import numpy as np
import math
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class KNN(object):
	"""docst
	ring for KNN"""
	def __init__(self, users, n_users, users_train, user_new, k, mf = None, dist_func = cosine_similarity):
		self.users = users
		self.n_users = n_users
		self.users_train = users_train	
		self.user_new = user_new.reshape(1, self.users_train.shape[1])
		self.k = k
		self.mf = mf
		self.dist_func = dist_func
	
	def nomarlize_users(self):
		self.user_new = (self.user_new - np.min(self.users_train, axis=0))/np.ptp(self.users_train, axis = 0)
		self.users_train = (self.users_train - np.min(self.users_train, axis=0))/np.ptp(self.users_train, axis = 0)

	def similarity(self):
		self.S = self.dist_func(self.user_new, self.users_train)

	def refresh(self):
		"""
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
		self.nomarlize_users()
		self.similarity()

	def fit(self):
		self.refresh()

	def __caculate_feature_user_new(self):
        # find the k most similarity users
		nearest_ids = np.argsort(self.S[0])[-self.k:] 
		# and the corresponding similarity levels
		nearest_s = self.S[0, nearest_ids]
        # How did each of 'near' users rated item i
		nearest_s = np.reshape(nearest_s, (1, self.k))
		tich = self.mf.W.T[nearest_ids].T*nearest_s
		print("tich: \n", tich)
		feature_user_new = (tich.sum(axis=1))/(np.abs(nearest_s).sum() + 1e-8)
		return feature_user_new
    
	def pred(self, mf_k = 10):
		feature_user_new = self.__caculate_feature_user_new()
		feature_user_new = np.reshape(feature_user_new, (mf_k,1))
		rate_pred = self.mf.X.dot(feature_user_new) 
		print("rating:\n",rate_pred)
		best_rate_ids = np.argsort(rate_pred[:, 0])[-10:]
		print("best rate:\n", best_rate_ids )
