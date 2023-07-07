import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score

bike_data = pd.read_csv('data/daily-bike-share.csv')

# lets make a new column which can tell which day it was on each date

bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day

# features
numeric_features = ['holiday','weekday','workingday']
# label
label = bike_data['rentals']

# # create a figure for 2 subplots (2 rows,1 column)
# fig,ax = plt.subplots(2,1,figsize=(9,12))

# # plot the histogram
# ax[0].hist(label,bins=100)
# ax[0].set_ylabel('Frequency')
# # add the lines for the mean, median and mode
# ax[0].axvline(label.mean(), color='magenta', linestyle='dashed',linewidth=2)
# ax[0].axvline(label.median(),color='cyan',linestyle='dashed',linewidth=2)

# # plot the boxplot
# ax[1].boxplot(label,vert=False)
# ax[1].set_xlabel('Rentals')

# fig.suptitle('Rental Distribution')

# plt.show()
# fig.show()


# for col in numeric_features:
#     fig = plt.figure(figsize=(9,6))
#     ax = fig.gca()
#     feature = bike_data[col]
#     feature.hist(bins=100, ax=ax)
#     ax.axvline(feature.mean(),color='magenta',linestyle='dashed',linewidth=2)
#     ax.axvline(feature.median(),color='cyan',linestyle='dashed',linewidth=2)
#     ax.set_title(col)
# plt.show()


# categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit','day']
# for col in categorical_features:
#     counts = bike_data[col].value_counts().sort_index()
#     fig = plt.figure(figsize=(9,6))
#     ax = fig.gca()
#     counts.plot.bar(ax=ax,color='steelblue')
#     ax.set_title(col + 'counts')
#     ax.set_xlabel(col)
#     ax.set_ylabel('Frequency')
# plt.show()


# for col in numeric_features:
#     fig = plt.figure(figsize=(9,6))
#     ax= fig.gca()
#     feature = bike_data[col]
#     label = bike_data['rentals']
#     correlation = feature.corr(label)
#     plt.scatter(x=feature,y=label)
#     plt.xlabel(col)
#     plt.ylabel('Bike Rentals')
#     ax.set_title('rentals vs '+ col + '- correlation: ' + str(correlation))
# plt.show()



# Separate features and labels
features = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values
label =  bike_data['rentals'].values
# print('Features:',features[:10], '\nLabels:', label[:10], sep='\n')

features_train,features_test,label_train,label_test = train_test_split(features,label,test_size=.30,random_state=00)
print('Training set: %d rows\nTest set: %d rows' % (features_train.shape[0],features_test.shape[0]))

# we will use linear regression method and fit that on a training set
model = LinearRegression().fit(features_train,label_train)
# model will predict the features 
predictions = model.predict(features_test)
np.printoptions(suppress=True)
print('Predicted labels: ', np.round(predictions)[:10])
print('Actual labels: ', label_test[:10])

plt.scatter(label_test,predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')

z = np.polyfit(label_test,predictions,1)
p = np.poly1d(z)
plt.plot(label_test,p(label_test),color='magenta')
plt.show()

mse = mean_squared_error(label_test,predictions)
print("MSE: ",mse)

rmse = np.sqrt(mse)
print("RMSE: ", rmse)

r2 = r2_score(label_test,predictions)
print("R2: ",r2)
