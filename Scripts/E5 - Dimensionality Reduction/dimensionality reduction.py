
import numpy as np
from numpy import *
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
#from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.decomposition import *
from sklearn.lda import LDA
from sklearn import metrics

train_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-2/User-8/1-8-2-1-GI.csv", dtype=float, delimiter=",")
test_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-2/User-8/1-8-2-2-GI.csv", dtype=float, delimiter=",")

target_train = np.ones(len(train_data))
row = 0
while row < len(train_data):
    if np.any(train_data[row, 0:3] != [1, 8, 2]):
        target_train[row] = 0
    row += 1

row = 0
target_test = np.ones(len(test_data))
while row < len(test_data):
    if np.any(test_data[row, 0:3] != [1, 8, 2]):
        target_test[row] = 0
    row += 1

sample_train = train_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17,18,19]]
sample_test = test_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17,18,19]]

"""
i = 0
while i <= 14:
    sample_train[:, i] = [log(x) for x in train_data[:, i]]
    sample_test[:, i] = [log(x) for x in test_data[:, i]]
    i += 1
"""
scaler = preprocessing.MinMaxScaler().fit(sample_train)
sample_train_scaled = scaler.transform(sample_train)
sample_test_scaled = scaler.transform(sample_test)

clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(sample_train, target_train)

prediction = clf.predict(sample_test)
print(metrics.confusion_matrix(target_test, prediction))
print(metrics.roc_auc_score(target_test, prediction))

pca = PCA(n_components=5)
sample_train_pca = pca.fit(sample_train_scaled).transform(sample_train_scaled)
sample_test_pca = pca.transform(sample_test_scaled)

scaler = preprocessing.MinMaxScaler().fit(sample_train_pca)
sample_train_scaled_pca = scaler.transform(sample_train_pca)
sample_test_scaled_pca = scaler.transform(sample_test_pca)

clf.fit(sample_train_pca, target_train)

prediction = clf.predict(sample_test_pca)
print(metrics.confusion_matrix(target_test, prediction))
print(metrics.roc_auc_score(target_test, prediction))



"""
inverse = pca.inverse_transform(sample_train_pca)
clf.fit(inverse, target_train)
prediction = clf.predict(sample_test_scaled)
print(metrics.confusion_matrix(target_test, prediction))
print(metrics.roc_auc_score(target_test, prediction))

ipca = FastICA(n_components=7)
sample_train_ipca = ipca.fit(sample_train_scaled).transform(sample_train_scaled)
sample_test_ipca = ipca.transform(sample_test_scaled)
clf.fit(sample_train_ipca, target_train)

prediction = clf.predict(sample_test_ipca)
print(metrics.confusion_matrix(target_test, prediction))
print(metrics.roc_auc_score(target_test, prediction))

inverse = ipca.inverse_transform(sample_train_ipca)
clf.fit(inverse, target_train)
prediction = clf.predict(sample_test_scaled)
print(metrics.confusion_matrix(target_test, prediction))
print(metrics.roc_auc_score(target_test, prediction))

lda = LDA(n_components=7)
sample_train_lda = lda.fit(sample_train_scaled, target_train).transform(sample_train_scaled)
sample_test_lda = lda.fit(sample_train_scaled, target_train).transform(sample_test_scaled)
clf.fit(sample_train_lda, target_train)

prediction = clf.predict(sample_test_lda)
print(metrics.confusion_matrix(target_test, prediction))
print(metrics.roc_auc_score(target_test, prediction))
"""