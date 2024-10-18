from sklearn.datasets import load_iris

data = load_iris()
type(data)
X = data.data #features
Y = data.target #target
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2) #30% of the data will be used for testing
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

print(Y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test,Y_pred)
print("Accuracy: {0:.2f}".format(acc))

accuracy=[]
for i in range (5,31):
  print
  val=model = KNeighborsClassifier(n_neighbors=i)
  val.fit(X_train,Y_train)
  Y_p = model.predict(X_test)
  ele = accuracy_score(Y_test,Y_pred)
  accuracy.append(ele)

x_array=[]
for i in range(5,31):
  x_array.append(i)

# %Matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
x=np.array(x_array)
y=np.array(accuracy)
plt.plot(x, y)
plt.show()


