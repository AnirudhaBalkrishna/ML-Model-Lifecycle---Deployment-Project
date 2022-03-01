#Imports
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

#Data Pre-processing
dataset = pd.read_csv('data.csv')

status_map = {'Admit': 1, 'Reject': 0}
dataset['status'] = dataset['status'].map(status_map)

eng_test_map = {'TOEFL': 1, 'IELTS': 2}
dataset['eng_test'] = dataset['eng_test'].map(eng_test_map)

y = dataset.pop('status')
X = dataset.drop(['university_name'], axis = 1)

#Classifier
classifier = RandomForestClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X, y)
joblib.dump(classifier, 'classifier.pkl')

#Scalar
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
joblib.dump(sc, 'scalar.pkl')

#Recommender
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
recommender = knn.fit(X_scaled)
joblib.dump(recommender, 'recommender.pkl')