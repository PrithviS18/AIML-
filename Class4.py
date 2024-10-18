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
