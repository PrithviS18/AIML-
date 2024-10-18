import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# We need to give separator and header
df4 = pd.read_csv('/content/Q4_data4.txt',sep=',',header=None)
df4.head()

#We use inPlace for permanent change
df4.rename(columns={0:'Size in Squared feet', 1:'Zone',2:'Price in $'},inplace=True)

x = df4['Size in Squared feet']
y = df4['Price in $']

m = y.size
print(m)
xbar = np.mean(x)
ybar = np.mean(y)

num=0
den=0
for i in range(0,m):
  num = num + (x[i]-xbar)*(y[i]-ybar)
  den = den +(x[i]-xbar)**2
theta1=(num/den)
theta0 = ybar - (theta1*xbar)

print(theta1)
print(theta0)

y_pred=[]
for i in range (m):
  y_pred.append((theta1*x[i])+theta0)
print(y_pred)

plt.figure(figsize=(10,7))
plt.scatter(x,y,color='blue',label='Original Data') #first independent variable then dependent variable
plt.plot(x,y_pred,color='red',label='Predicted Line')
plt.xlabel('Size in ft2')
plt.ylabel('Price in $')

num1=0
den1 =0
for i in range (m):
  num1 = num1 + (y_pred[i] - ybar)**2
  den1 = den1 + (y[i] - ybar)**2
r2score = num1/den1
print (r2score)

r2score = r2score *100
print (r2score)

#Verifying using Scientific Tool Kit
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.2) #20% data will be for test data


X_train = X_train.values.reshape(-1,1) #coming in terms of array

Y_test = Y_test.values.reshape(-1,1)
Y_train = Y_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
my_lr = LinearRegression()
my_lr.fit(X_train,Y_train)

my_lr.coef_
my_lr.intercept_

from sklearn.metrics import r2_score, mean_squared_error
Y_pred_train = my_lr.predict(X_train)
Y_pred_test= my_lr.predict(X_test)
r2_score(Y_train,Y_pred_train)
r2_score(Y_test,Y_pred_test)
