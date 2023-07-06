import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# read the file from data folder
df_flights = pd.read_csv('data/flights.csv',delimiter=',',header='infer')
# fill the empty cells with value
df_flights['DepDel15'] = df_flights['DepDel15'].fillna(0)

# print(df_flights.isnull().sum())
groups = df_flights.groupby(df_flights['ArrDel15'] == 1)

flights_delayed = 0
flight_ontime = 0

for group in groups:
    if group[0] == True:
        flights_delayed = len(group[1])
    else:
        flight_ontime = len(group[1])



print('Flights on time: {}\nFlights delayed: {}'.format(flight_ontime,flights_delayed))
print(df_flights['ArrDel15'].mean())

delay_counts = df_flights['ArrDel15'].value_counts()
plt.pie(delay_counts,labels=delay_counts,autopct='%1.1f%%')
plt.title('Flights get delayed by airports')
plt.show()

# plt.pie(x=df_flights['ArrDel15'],height=len(df_flights))