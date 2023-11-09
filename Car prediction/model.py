import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

car_ds = pd.read_csv('car_data.csv')
car_ds.drop("User ID", axis=1, inplace=True)

car_ds['Gender'] = car_ds['Gender'].map({'Male': 1, 'Female': 0})

X = car_ds.drop(columns='Purchased', axis=1)
Y = car_ds['Purchased']

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
X = standardized_data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

Classifier = svm.SVC(kernel='linear')
Classifier.fit(X_train, Y_train)

pickle.dump(Classifier, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
