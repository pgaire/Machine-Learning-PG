# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Reading the data
train_data = pd.read_csv('./train_final.csv')
test_data = pd.read_csv('./test_final.csv')



# train_data.replace("?", np.nan, inplace = True)
# test_data.replace("?", np.nan, inplace = True)
# # Checking if there exist null values in the dataset

# Data cleaning

attributes=['age','workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']
target=['income>50K']

# Assigining attributes and target data
x_train=train_data[attributes]
y_train=train_data[target]
x_test=test_data[attributes]


x_train['education'].value_counts()[:20].plot(kind='bar', figsize=(7, 6), rot=0)
#x_train['occupation'].value_counts()[:20].plot(kind='bar')
#x_train['workclass'].value_counts()[:20].plot(kind='bar')

category=[ 'workclass',  'education', 
       'marital.status', 'occupation', 'relationship', 'race', 'sex',
        'native.country']
for x in category:
    x_train[x] = LabelEncoder().fit_transform(x_train[x])
    x_test[x] = LabelEncoder().fit_transform(x_test[x])

# for x in category:
#     x_train[x].apply(LabelEncoder().fit_transform)
#     x_test[x].apply(LabelEncoder().fit_transform)
    


print(x_test.shape)




# Model training decision tree
Income_classifier1 = DecisionTreeClassifier(random_state=0)
Income_classifier1.fit(x_train, y_train)

# Prediction
Income_prediction1 = Income_classifier1.predict(x_test)

# # Checking accuracy
np.savetxt('prediction1.csv', Income_prediction1,delimiter='\t')



# Model training random forest
Income_classifier2 = RandomForestClassifier(n_estimators=100, max_depth=20)
Income_classifier2.fit(x_train, y_train)

# Prediction
Income_prediction2 = Income_classifier2.predict_proba(x_test)

# # Checking accuracy
np.savetxt('prediction2.csv', Income_prediction2,delimiter='\t')

# confusion matrix
y_predict = Income_classifier2.predict(x_train)
conf_matrix = confusion_matrix(y_train, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(2)
plt.show()

# Model training logistic regression
Income_classifier3 = KNeighborsClassifier(n_neighbors=5)
Income_classifier3.fit(x_train, y_train)
# Prediction
Income_prediction3 = Income_classifier3.predict(x_test)
# # Checking accuracy
np.savetxt('prediction3.csv', Income_prediction3,delimiter='\t')

# Model training logistic regression
Income_classifier4 = LogisticRegression()
Income_classifier4.fit(x_train,y_train)
# Prediction
Income_prediction4 = Income_classifier4.predict(x_test)
# # Checking accuracy
np.savetxt('prediction4.csv', Income_prediction4,delimiter='\t')

# Model training with XGBclassifier
Income_classifier5 = XGBClassifier(random_state=1,learning_rate=0.001)
Income_classifier5.fit(x_train,y_train)
# Prediction
Income_prediction5 = Income_classifier5.predict(x_test)
# # Checking accuracy
np.savetxt('prediction5.csv', Income_prediction5,delimiter='\t')

# Model training with SVM
Income_classifier6 = SVC()
Income_classifier6.fit(x_train,y_train)
# Prediction
Income_prediction6 = Income_classifier6.predict(x_test)
# # Checking accuracy
np.savetxt('prediction6.csv', Income_prediction6,delimiter='\t')



