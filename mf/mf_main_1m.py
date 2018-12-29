# Import packages
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from mf import MF
import pickle
# Define file directories
MOVIELENS_DIR = '../dataset/ml-1m'
USER_DATA_FILE = 'users.dat'
MOVIE_DATA_FILE = 'movies.dat'
RATING_DATA_FILE = 'ratings.dat'
# Specify User's Age and Occupation Column
AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }
# Define csv files to be saved into
USERS_CSV_FILE = 'users.csv'
MOVIES_CSV_FILE = 'movies.csv'
RATINGS_CSV_FILE = 'ratings.csv'
# Read the Ratings File
ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Set max_userid to the maximum user_id in the ratings
max_userid = ratings['user_id'].drop_duplicates().max()
# Set max_movieid to the maximum movie_id in the ratings
max_movieid = ratings['movie_id'].drop_duplicates().max()

# Process ratings dataframe for Keras Deep Learning model
# Add user_emb_id column whose values == user_id - 1
# ratings['user_emb_id'] = ratings['user_id'] - 1
# Add movie_emb_id column whose values == movie_id - 1
# ratings['movie_emb_id'] = ratings['movie_id'] - 1
ratings_base = ratings.as_matrix()
ratings_base[:, :2] -= 1

rate_train, rate_test= train_test_split(ratings_base, test_size=0.3)

print (rate_train.shape)
print (rate_test.shape)
# Read the Users File
users = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
# users['age_desc'] = users['age'].apply(lambda x: AGES[x])
# users['occ_desc'] = users['occupation'].apply(lambda x: OCCUPATIONS[x])
# print (len(users), 'descriptions of', max_userid, 'users loaded.')
user_train = users.as_matrix()
all_users = user_train[:, :4]
all_users[all_users[:,1]=='M', 1] = 1
all_users[all_users[:,1]=='F', 1] = 0
print(all_users)



rs = MF(rate_train, K = 100, lam = .1, print_every = 10,learning_rate = 0.75, max_iter = 100, user_based = 1)
# print("X0:\n", rs.X)
# print("rate_test:\n", rate_test)
# in_file = open("MF.obj", "rb") # opening for [r]eading as [b]inary
# rs = pickle.load(in_file) # if you only wanted to read 512 bytes, do .read(512)
# in_file.close()
# print(type(rs))

rs.fit()

file_mf = open('MF_1m.obj', 'wb')
pickle.dump(rs, file_mf)
file_mf.close()
print(type(rs))
# print("X1:\n", rs.X)
print("utility:\n", rs.X.dot(rs.W) + rs.mu)
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print ('\nUser-based MF, RMSE =', RMSE)

# # Read the Movies File
# movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE), 
#                     sep='::', 
#                     engine='python', 
#                     encoding='latin-1',
#                     names=['movie_id', 'title', 'genres'])
# print (len(movies), 'descriptions of', max_movieid, 'movies loaded.')
# print(movies)
