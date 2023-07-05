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

def show_distribution(var_data):
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]
    print('Minimum value: {:.2f}\nMaximum value: {:.2f}\nMedian value: {:.2f}\nMean value: {:.2f}\nMode value: {:.2f}'.format(min_val,max_val,mean_val,med_val,mod_val))

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # Add a title to the Figure
    fig.suptitle('Data Distribution')

    # Show the figure
    plt.show()
    fig.show()



# show_distribution(df_students['Grade'])

# show_distribution(df_students['StudyHours'])

col = df_students[df_students['StudyHours'] > 1]['StudyHours']
# show_distribution(col)
q01 = df_students['StudyHours'].quantile(0.01)
col = df_students[df_students['StudyHours'] > q01]['StudyHours']



def show_density(var_data):
    fig = plt.figure(figsize=(10,4))
    # plot density
    var_data.plot.density()


    # add titles and labels
    plt.title("Data Density")

    # show the mean,median and mode 
    plt.axvline(var_data.mean(), color = 'cyan', linestyle='dashed',linewidth=2)
    plt.axvline(var_data.median(), color = 'red', linestyle='dashed',linewidth=2)
    plt.axvline(var_data.mode()[0], color = 'orange', linestyle='dashed',linewidth=2)

    plt.show()

# show_density(col)


for col_name in ['StudyHours','Grade']:
    col = df_students[col_name]
    range = col.max() - col.min()
    variance = col.var()
    standard_deviation = col.std()

    print('{}\n- Range: {:.2f}\n- Variance: {:.2f}\n- Standard Deviation: {:.2f}'.format(col_name,range,variance,standard_deviation))


# visualize the data on the graph
# we can add color, title, labels, grid and rotate the x markers inside the graph
# plt.bar(x=df_students['Name'],height=df_students['Grade'], color="orange")
# plt.title("Student Grades")
# plt.xlabel('Student')
# plt.ylabel('Grade')
# plt.grid(color="#95a5a6",linestyle='--', linewidth=2,axis='y',alpha=0.7)
# plt.show()

# fig,ax = plt.subplots(1,2,figsize=(10,4))
# create a bar plot of name vs grade on the first axis
# ax[0].bar(x=df_students['Name'], height=df_students['Grade'], color="orange")
# ax[0].set_title('Grades')
# ax[0].set_xticklabels(df_students['Name'], rotation=90)

# create a pie chart of pass counts
# pass_counts = df_students['Pass'].value_counts()
# ax[1].pie(pass_counts, labels=pass_counts)
# ax[1].set_title('Passing Grades')
# ax[1].legend(pass_counts.keys().tolist())

# fig.suptitle('Student Data')


# fig.show()

# get the variable to examine
# var_data = df_students['Grade']

# # create a figure
# fig = plt.figure(figsize=(10,4))

# #plot a histogram
# plt.hist(var_data)

#add the titles and labels
# plt.title('Data Distribution')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

# plt.show()
# fig.show()

# Measures of central tendency

# # #save grades in a variable to calculate median , mode , mean
# var_grades = df_students['Grade']

# # get statitics
# min_val = var_grades.min()
# max_val = var_grades.max()
# mean_val = var_grades.mean()
# med_val = var_grades.median()
# mod_val = var_grades.mode()[0]

#print out the statistics
# print('Minimum value: {:.2f}\nMaximum value: {:.2f}\nMean value: {:.2f}\nMedian value: {:.2f}\nMode value: {:.2f}\n'.format(min_val,max_val,mean_val,med_val,mod_val))

# # create a fig
# fig = plt.figure(figsize=(10,4))

# # plot a histogram

# plt.hist(var_grades)

# add lines for the statistics

# plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
# plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
# plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
# plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
# plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

# add title and labels

# plt.title("Data Distribution")
# plt.xlabel('Value')
# plt.ylabel('Frequency')

# plt.show()
# fig.show()


# fig = plt.figure(figsize=(10,4))

# plt.boxplot(var_grades)

# plt.title('Data Distribution')

# plt.show()
# fig.show()
