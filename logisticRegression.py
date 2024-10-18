import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# If text file then use sep and header
df = pd.read_csv('/content/Tumor_Database.csv')
df.head()

df.columns

x = df.loc[:,['radius_mean']]
y = df.loc[:,['diagnosis']]

df['diagnosis'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y = le.fit_transform(y)

y= pd.DataFrame(y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,Y_train)

y_pred_test = log_reg.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
accuracy_score(Y_test,y_pred_test)

y_pred_train = log_reg.predict(X_train)
accuracy_score(Y_train,y_pred_train)

f1_score(Y_test,y_pred_test)
f1_score(Y_train,y_pred_train)

precision_score(Y_train,y_pred_train)
precision_score(Y_test,y_pred_test)

recall_score(Y_test,y_pred_test)
recall_score(Y_train,y_pred_train)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y_test,y_pred_test,labels=log_reg.classes_)
cm

ConfusionMatrixDisplay(cm,display_labels=log_reg.classes_).plot()
