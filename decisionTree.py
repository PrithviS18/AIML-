from sklearn.datasets import load_iris

data = load_iris()
type(data)
X = data.data #features
Y = data.target #target
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2) #30% of the data will be used for testing
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test,Y_pred)
print("Accuracy: {0:.2f}".format(acc))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(8,6))
plot_tree(model,
          feature_names = data.feature_names,
          class_names = data.target_names,
          filled=True)
plt.show()
