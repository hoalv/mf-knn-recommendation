from mf import MF
import pandas as pd
import pickle

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('../dataset/ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('../dataset/ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

rs = MF(rate_train, K = 10, lam = .1, print_every = 10,learning_rate = 0.75, max_iter = 100, user_based = 1)
# print("X0:\n", rs.X)
# print("rate_test:\n", rate_test)
# in_file = open("MF.obj", "rb") # opening for [r]eading as [b]inary
# rs = pickle.load(in_file) # if you only wanted to read 512 bytes, do .read(512)
# in_file.close()
# print(type(rs))

rs.fit()

file_mf = open('MF.obj', 'wb')
pickle.dump(rs, file_mf)
file_mf.close()
print(type(rs))
# print("X1:\n", rs.X)
print("utility:\n", rs.X.dot(rs.W) + rs.mu)
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print ('\nUser-based MF, RMSE =', RMSE)