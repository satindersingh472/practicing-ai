import pandas as pd
import numpy as np

# read the file from data folder
df_flights = pd.read_csv('data/flights.csv',delimiter=',',header='infer')
# fill the empty cells with value
df_flights['DepDel15'] = df_flights['DepDel15'].fillna(0)

# print(df_flights.isnull().sum())

print(df_flights)