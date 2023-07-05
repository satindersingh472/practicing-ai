import numpy as np
import pandas as pd

# an array of grades 
data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]

#modifiy an array into numpy array so that we can mathematical calculations easily
grades = np.array(data)

#with shape we can have an idea about the dimensions of an array
grades.shape

# array.mean() can help find the mean of an array

# an array of time devoted to study
study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]

# create 2d array 
student_data = np.array([study_hours,grades])
# print(student_data.shape)
# array can be accessed with regular postion method 
# print(student_data[0][0])

avg_study = student_data[0].mean()
avg_grade = student_data[1].mean()

# print('Average study hours: {:.2f}\n Average grade {:.2f}'. format(avg_study,avg_grade)  )

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
                                     'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
                                     'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
                            'StudyHours':student_data[0],
                            'Grade':student_data[1]})

# print(df_students[df_students['Name'] == 'Rajab'])
# print(df_students.query('Name == "Rajab"'))

df_students = pd.read_csv('grades.csv', delimiter=',',header='infer')
# print(df_students.head())

#handling missing values in data
# print(df_students[df_students.isnull().any(axis =1)])
#after getting empty values we can replace missing values with mean value of an array

df_students['StudyHours'] = df_students['StudyHours'].fillna(df_students['StudyHours'].mean())

# print(df_students)

#drop the rows with null values
df_students = df_students.dropna(axis=0,how='any')
# print(df_students)

#get the mean of study hours using the column name as an index
mean_study = df_students['StudyHours'].mean()
# print(mean_study)

#get the mean of grades using the column name as a property
mean_grade = df_students.Grade.mean()
# print(mean_grade)


# print the mean study and grades

# print('Average weekly study hours: {:.2f}\nAverage Grade: {:.2f}'.format(mean_study,mean_grade))

# get the students who studied more than average

# print(df_students[df_students['StudyHours'] > mean_study])

# get the mean  grade of students who scored more than average

# print(df_students[df_students['StudyHours'] > mean_study]['Grade'].mean())
# to add another column name pass
passes = pd.Series(df_students['Grade'] >= 60)
# concat the column with the array on the same axis
df_students = pd.concat([df_students,passes.rename('Pass')],axis=1)
# print(df_students)

# grouping student based on pass and fail  and calculating mean of students grades and hours they studied

print(df_students.groupby(df_students['Pass'])[['Grade','StudyHours']].mean())