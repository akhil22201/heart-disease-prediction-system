# Heart Disease Prediction Using Machine Learning Algorithms

# ## Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("successful import ;)")

# ## Read Dataset and preprocess the data

heartData = pd.read_csv(r"C:\Users\akhil\Desktop\Project Work\Heart Disease Prediction System\dataset\heart_data_set.csv")

heartData.info()
heartData.describe()

# Total missing percent of data
missing_data= heartData.isnull().sum()
total_percentage = (missing_data.sum()/heartData.shape[0]) * 100
print(f'Total percentage of missing data is {round(total_percentage,2)}%')

duplicate=heartData[heartData.duplicated()]
print("Duplicate rows:")
print(duplicate)
heartData=heartData.drop_duplicates()

# Correlation matrix (optional)
rcParams['figure.figsize'] = 10,10
plt.matshow(heartData.corr())
plt.yticks(np.arange(heartData.shape[1]), heartData.columns)
plt.xticks(np.arange(heartData.shape[1]), heartData.columns)
plt.colorbar()

corr = heartData.corr()
# corr.style.background_gradient(cmap='coolwarm') # Optional for notebook only

# Count of each target class
rcParams['figure.figsize'] = 8,6
plt.bar(heartData['target'].unique(), heartData['target'].value_counts(), color = ['black', 'silver'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')

# Divide data into training/testing
X = heartData.drop(['target'], axis = 1)
y = heartData['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
print("XTrain->",X_train.shape[0],"XTest->",X_test.shape[0], "YTrain->",y_train.shape[0], "XTrain->",y_test.shape[0])

# -------------------------
# Model Building
# -------------------------

# KNN Algorithm
knn_scores = []
for k in range(2,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train.values, y_train.values)
    knn_score=round(knn_classifier.score(X_test.values, y_test.values),2)
    knn_scores.append(knn_score)

knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(X_train, y_train)
knn_score=knn_classifier.predict(X_test)
print(classification_report(y_test,knn_score))

plt.plot([k for k in range(2, 21)], knn_scores, color = 'red')
for i in range(2,21):
    plt.text(i, knn_scores[i-2], (i, knn_scores[i-2]))
plt.xticks([i for i in range(2, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('KNN Scores for different K neighbours')

print(knn_scores)
print(max(knn_scores))

knn_classifier = KNeighborsClassifier(n_neighbors = 11)
knn_classifier.fit(X_train.values, y_train.values)
check_data_by_sudhansu = np.array([[52,1,0,125,212,0,1,168,0,1,2,2,3]])
prediction_result = knn_classifier.predict(check_data_by_sudhansu)
print("Prediction: {}".format(prediction_result))

# Support Vector Machine
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train.values, y_train.values)
    svc_scores.append(round(svc_classifier.score(X_test.values, y_test.values),2))

svc_classifier = SVC(kernel = kernels[0])
svc_classifier.fit(X_train.values, y_train.values)
svc_prediction_result=svc_classifier.predict(X_test.values)
print(accuracy_score(y_test.values,svc_prediction_result))

colors = rainbow(np.linspace(0, 1, len(kernels)))
plt.bar(kernels, svc_scores, color = colors)
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('SVM scores Activation function wise...')

# Decision Tree
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train.values, y_train.values)
    dt_scores.append(round(dt_classifier.score(X_test.values, y_test.values),2))
print("Done")
print(dt_scores)

dt_classifier = DecisionTreeClassifier(max_features = 13, random_state = 0)
dt_classifier.fit(X_train.values, y_train.values)

plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')

# Random Forest
rf_scores = []
estimators = [10, 20,100, 200, 500]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train.values, y_train.values)
    rf_scores.append(round(rf_classifier.score(X_test.values, y_test.values),2))

colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')

rf_model = RandomForestClassifier(n_estimators = 10, random_state = 0)
rf_model.fit(X_train.values, y_train.values)
rf_model_result=rf_classifier.predict(X_test.values)
print(accuracy_score(y_test,rf_model_result))

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train.values, y_train.values)
logistic_model_prediction=logistic_model.predict(X_test.values)
print(accuracy_score(y_test.values,logistic_model_prediction))
print(classification_report(y_test.values,logistic_model_prediction))

# Save trained models
import pickle
all_models=[rf_model,logistic_model,dt_classifier,svc_classifier,knn_classifier]

with open("models.pkl", 'wb') as files:
    pickle.dump(all_models, files)
print("Done")

# Load models to verify
open_file = open("models.pkl", "rb")
loaded_list = pickle.load(open_file)
print(loaded_list)
open_file.close()
print("Done")
