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
# we can add color, title, labels, grid and rotate the x markers inside the graph
plt.bar(x=df_students['Name'],height=df_students['Grade'], color="orange")
plt.title("Student Grades")
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color="#95a5a6",linestyle='--', linewidth=2,axis='y',alpha=0.7)
plt.show()

fig,ax = plt.subplots(1,2,figsize=(10,4))
# create a bar plot of name vs grade on the first axis
ax[0].bar(x=df_students['Name'], height=df_students['Grade'], color="orange")
ax[0].set_title('Grades')
ax[0].set_xticklabels(df_students['Name'], rotation=90)

# create a pie chart of pass counts
pass_counts = df_students['Pass'].value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())

fig.suptitle('Student Data')


fig.show()

# print(df_students)