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
# plt.show()

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


# fig.show()

# get the variable to examine
var_data = df_students['Grade']

# create a figure
fig = plt.figure(figsize=(10,4))

#plot a histogram
plt.hist(var_data)

#add the titles and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# plt.show()
# fig.show()

# Measures of central tendency

#save grades in a variable to calculate median , mode , mean
var_grades = df_students['Grade']

# get statitics
min_val = var_grades.min()
max_val = var_grades.max()
mean_val = var_grades.mean()
med_val = var_grades.median()
mod_val = var_grades.mode()[0]

#print out the statistics
print('Minimum value: {:.2f}\nMaximum value: {:.2f}\nMean value: {:.2f}\nMedian value: {:.2f}\nMode value: {:.2f}\n'.format(min_val,max_val,mean_val,med_val,mod_val))

# create a fig
fig = plt.figure(figsize=(10,4))

# plot a histogram

plt.hist(var_grades)

# add lines for the statistics

plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

# add title and labels

plt.title("Data Distribution")
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.show()
fig.show()