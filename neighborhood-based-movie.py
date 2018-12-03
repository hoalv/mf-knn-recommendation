from cf import CF
import pandas as pd 
r_cols = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('ex.dat', sep = ' ', names = r_cols, encoding='latin-1')
Y_data = ratings.as_matrix()


# user-user cf
rs = CF(Y_data, k = 3, uuCF = 1)
rs.fit()

rs.print_recommendation()
print("sim: \n", rs.S)

# item-item cf
# rs = CF(Y_data, k = 2, uuCF = 0)
# rs.fit()

# rs.print_recommendation()
