import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# import different models
from sklearn.tree import DecisionTreeRegressor,export_text
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#load the training set
bike_data = pd.read_csv('data/daily-bike-share.csv')
bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day
numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']
bike_data[numeric_features + ['rentals']].describe()
# print(bike_data.head())

# seperate features and labels
# we have two np arrays - features and label
features = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values
label = bike_data['rentals'].values

# split the data between testing and training sets
features_train,features_test,label_train,label_test = train_test_split(features,label,test_size=.30,random_state=0)

# print('Training set: %d rows\nTesting set: %d rows' % (features_train.shape[0],features_test.shape[0]))

# fit a model on the training set
model = Lasso().fit(features_train,label_train)

# evaluate the model using the test data
# according to my point of view predictions will have the values the model predicted after seeing the features 
#  that were kept to test and after predicting we will calculate the mean squared error
predictions = model.predict(features_test)
# calculate the mean squared error after doing predictions, we have the actual test label values and prediction and 
# we will calculate the mse between those two arrays of values
print('Linear Regression model \n---------------------------------')
mse = mean_squared_error(label_test,predictions)
print("MSE:",mse)
# root mean squared error
rmse = np.sqrt(mse)
print("Root Mean Squared Error: ",rmse)
# r2 R-Squared
r2 = r2_score(label_test,predictions)
print('R2: ',r2)


# plot predicted vs actual

plt.scatter(label_test,predictions)
plt.xlabel('Actual Test Labels')
plt.ylabel("Predicted Test Labels")
plt.title('Daily Bike Rentals Predictions - Linear Regression ')

# overlay the regression line
z = np.polyfit(label_test,predictions,1)
p = np.poly1d(z)
plt.plot(label_test,p(label_test),color='magenta')
plt.show()


# Decision Tree algorithm
#train the model
model = DecisionTreeRegressor().fit(features_train,label_train)
# visualize the model tree
tree = export_text(model)

# evaluate the model using test data
predictions = model.predict(features_test)
print('Decision Tree \n--------------------------')
mse = mean_squared_error(label_test,predictions)
print("MSE:",mse)
# root mean squared error
rmse = np.sqrt(mse)
print("Root Mean Squared Error: ",rmse)
# r2 R-Squared
r2 = r2_score(label_test,predictions)
print('R2: ',r2)

# plot predicted vs actual


plt.scatter(label_test,predictions)
plt.xlabel('Actual Test Labels')
plt.ylabel("Predicted Test Labels")
plt.title('Daily Bike Rentals Predictions - Decision Tree Regressor ')

# overlay the regression line
z = np.polyfit(label_test,predictions,1)
p = np.poly1d(z)
plt.plot(label_test,p(label_test),color='magenta')
plt.show()

# train the model
model = RandomForestRegressor().fit(features_train,label_train)

#Evaluate the model using the test data
predictions = model.predict(features_test)
print('Random Forest Regressor model \n--------------------------')
mse = mean_squared_error(label_test,predictions)
print("MSE:",mse)
# root mean squared error
rmse = np.sqrt(mse)
print("Root Mean Squared Error: ",rmse)
# r2 R-Squared
r2 = r2_score(label_test,predictions)
print('R2: ',r2)

# plot predicted vs actual


plt.scatter(label_test,predictions)
plt.xlabel('Actual Test Labels')
plt.ylabel("Predicted Test Labels")
plt.title('Daily Bike Rentals Predictions - Random Forest Regressor ')

# overlay the regression line
z = np.polyfit(label_test,predictions,1)
p = np.poly1d(z)
plt.plot(label_test,p(label_test),color='magenta')
plt.show()

# Gradient Boosting Regressor

#train the model
model = GradientBoostingRegressor().fit(features_train,label_train)

#evaluate model
predictions = model.predict(features_test)
print('GradientBoostReggressor ---------------')
mse = mean_squared_error(label_test,predictions)
print("MSE:",mse)
# root mean squared error
rmse = np.sqrt(mse)
print("Root Mean Squared Error: ",rmse)
# r2 R-Squared
r2 = r2_score(label_test,predictions)
print('R2: ',r2)

plt.scatter(label_test,predictions)
plt.xlabel('Actual Test Labels')
plt.ylabel("Predicted Test Labels")
plt.title('Daily Bike Rentals Predictions - Gradient Boost Regressor ')

# overlay the regression line
z = np.polyfit(label_test,predictions,1)
p = np.poly1d(z)
plt.plot(label_test,p(label_test),color='magenta')
plt.show()
