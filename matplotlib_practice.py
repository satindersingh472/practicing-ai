import pandas as pd
from matplotlib import pyplot as plt


#load the data from csv file
df_students = pd.read_csv('grades.csv', delimiter=',',header='infer')
# remove any rows with missing data
df_students = df_students.dropna(axis=0,how='any')
# create a series of students got grade 60 or more
passes = pd.Series(df_students['Grade'] >= 60)
# concat the pass series at axis 1 
df_students = pd.concat([df_students,passes.rename('Pass')],axis=1)
# visualize the data on the graph
plt.bar(x=df_students['Name'],height=df_students['Grade'])
plt.show()

# print(df_students)