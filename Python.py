import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import seaborn as sns
%matplotlib inline
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline

df = pd.read_csv("housing.csv")
rows, colums = df.shape

housingMedian0 = df.iloc[0]["housing_median_age"]
housingMedian1= df.iloc[1]["housing_median_age"]

totalRooms0 = df.iloc[0]["total_rooms"]
totalRooms1= df.iloc[1]["total_rooms"]

totalBedrooms0 = df.iloc[0]["total_bedrooms"]
totalBedrooms1= df.iloc[1]["total_bedrooms"]

#print(housingMedian0+housingMedian1)
#print(totalRooms0+totalRooms1)
#print(totalBedrooms0+totalBedrooms1)

#plottwo = pd.read_csv("housing.csv")
#plotthree = pd.read_csv("housing.csv")
#plotfour = pd.read_csv("housing.csv")

#plottwo.plot(x='total_rooms', y='median_house_value')
#plotthree.plot(x='total_bedrooms', y='median_house_value')

#df.iloc[1:10].plot(x='housing_median_age', y='median_house_value')

#X = df.iloc[:,0:-1].values 
#y = df.iloc[:,-1].values

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=4)

#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

#y_pred_logistic = model_logistic.decision_function(X_test)
                        
##################################################################
HousingSet = pd.read_csv('housing.csv')
HousingSet.shape
HousingSet.describe()

HousingSet.plot(x='median_income', y='median_house_value', style='o')  
plt.title('Zillow Estimate-Kaggle:MedianIncome VS MedianHouseValue')  
plt.xlabel('median_income')  
plt.ylabel('median_house_value') 
plt.show()

X = HousingSet['median_income'].values.reshape(-1,1)
y = HousingSet['median_house_value'].values.reshape(-1,1)

#I have no idea what test_size does, I just played with it randomly (!)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

regr = LinearRegression()  
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.show()

print('MSE: ', metrics.mean_squared_error(y_test, y_pred))  
##################################################################
HousingSetTwo = pd.read_csv('housing.csv')
HousingSetTwo.shape
HousingSetTwo.describe()

HousingSetTwo.plot(x='total_rooms', y='median_house_value', style='o')  
plt.title('Zillow Estimate-Kaggle:Rooms VS MedianHouseValue')  
plt.xlabel('total_rooms')  
plt.ylabel('median_house_value') 
plt.show()

X = HousingSetTwo['total_rooms'].values.reshape(-1,1)
y = HousingSetTwo['median_house_value'].values.reshape(-1,1)

#I have no idea what test_size does, I just played with it randomly (!)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

regr = LinearRegression()  
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.show()

print('MSE: ', metrics.mean_squared_error(y_test, y_pred))  
##################################################################

