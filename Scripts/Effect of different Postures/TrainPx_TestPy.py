import numpy as np
from numpy import *
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import metrics

train_data = np.genfromtxt("../../Output Files/E6-Reduced Dimensional Dataset/posture-1/User-7/1-7-1-1-GI.csv", dtype=float, delimiter=",")
test_data = np.genfromtxt("../../Output Files/E6-Reduced Dimensional Dataset/posture-2/User-7/1-7-2-2-GI.csv", dtype=float, delimiter=",")

target_train = np.ones(len(train_data))
row = 0
while row < len(train_data):
    if np.any(train_data[row, 0:3] != [1, 7, 1]):
        target_train[row] = 0
    row += 1

row = 0
target_test = np.ones(len(test_data))
while row < len(test_data):
    if np.any(test_data[row, 0:3] != [1, 7, 2]):
        target_test[row] = 0
    row += 1

sample_train = train_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17]]
sample_test = test_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17]]
scaler = preprocessing.MinMaxScaler().fit(sample_train)
sample_train_scaled = scaler.transform(sample_train)
sample_test_scaled = scaler.transform(sample_test)

clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(sample_train_scaled, target_train)

prediction = clf.predict(sample_test_scaled)
auc = metrics.roc_auc_score(target_test, prediction)
print(auc)