
import numpy as np
from numpy import *
import os
from sklearn import svm
from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance

def outlier_Removal(percentage, posture):
    i_user = 1
    session = 1
    while i_user <= 31:
        while session <= 8:
            currentdirectory = os.getcwd()  # get the directory.
            parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
            path = parentdirectory + '\Output Files\E6-Outlier Removed Dataset/Posture-'+str(posture)+'/GenuineUser-'+str(i_user)+''
            if not os.path.exists(path):
                os.makedirs(path)

            data = np.genfromtxt("../../Output Files/E2-Genuine User-Session Split/Posture-"+str(posture)+"/GenuineUser-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(session)+".csv", dtype=float, delimiter=",")

            clf = svm.OneClassSVM(kernel='linear', nu= percentage)
            clf.fit(data)
            y = clf.predict(data)
            index = np.where(y != -1)
            RemovedOutliers = data[index[0], :]

            np.savetxt("../../Output Files/E6-Outlier Removed Dataset/Posture-"+str(posture)+"/GenuineUser-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(session)+".csv", RemovedOutliers, delimiter=",")
            session += 1
        session = 1
        i_user += 1

outlier_Removal(0.20, 2)

















"""
data1 = np.genfromtxt("../../Output Files/E2-Genuine User-Session Split/Posture-1/GenuineUser-6/1-6-1-2.csv", dtype=float, delimiter=",")
clf = svm.OneClassSVM(kernel='linear', nu= 0.20)
clf.fit(data1)
y = clf.predict(data1)
index = np.where(y != -1)
RemovedOutliers1 = data1[index[0], :]

pca = PCA(n_components=2)
data1_pca = pca.fit(data1).transform(data1)
RemovedOutliers1_pca = pca.fit(RemovedOutliers1).transform(RemovedOutliers1)
plt.scatter(data1_pca[:, 0], data1_pca[:, 1], color='g')
plt.scatter(RemovedOutliers1_pca[:, 0], RemovedOutliers1_pca[:, 1], color='r')
plt.show()

data2 = np.genfromtxt("../../Output Files/E2-Genuine User-Session Split/Posture-1/GenuineUser-12/1-12-1-2.csv", dtype=float, delimiter=",")
clf = svm.OneClassSVM(kernel='linear', nu= 0.20)
clf.fit(data2)
y = clf.predict(data2)
index = np.where(y != -1)
RemovedOutliers2 = data2[index[0], :]

pca = PCA(n_components=2)
data2_pca = pca.fit(data2).transform(data2)
RemovedOutliers2_pca = pca.fit(RemovedOutliers2).transform(RemovedOutliers2)
plt.scatter(data2_pca[:, 0], data2_pca[:, 1], color='g')
plt.scatter(RemovedOutliers2_pca[:, 0], RemovedOutliers2_pca[:, 1], color='r')
plt.show()

train_data = np.vstack((RemovedOutliers1, RemovedOutliers2))
#train_data = np.vstack((data1, data2))

data3 = np.genfromtxt("../../Output Files/E2-Genuine User-Session Split/Posture-1/GenuineUser-6/1-6-1-1.csv", dtype=float, delimiter=",")
clf = svm.OneClassSVM(kernel='linear', nu= 0.20)
clf.fit(data3)
y = clf.predict(data3)
index = np.where(y != -1)
RemovedOutliers3 = data3[index[0], :]

pca = PCA(n_components=2)
data3_pca = pca.fit(data3).transform(data3)
RemovedOutliers3_pca = pca.fit(RemovedOutliers3).transform(RemovedOutliers3)
plt.scatter(data3_pca[:, 0], data3_pca[:, 1], color='g')
plt.scatter(RemovedOutliers3_pca[:, 0], RemovedOutliers3_pca[:, 1], color='r')
plt.show()

data4 = np.genfromtxt("../../Output Files/E2-Genuine User-Session Split/Posture-1/GenuineUser-12/1-12-1-1.csv", dtype=float, delimiter=",")
clf = svm.OneClassSVM(kernel='linear', nu= 0.20)
clf.fit(data4)
y = clf.predict(data4)
index = np.where(y != -1)
RemovedOutliers4 = data4[index[0], :]

test_data = np.vstack((RemovedOutliers3, RemovedOutliers4))
#test_data = np.vstack((data3, data4))

target_train = np.ones(len(train_data))
row = 0
while row < len(train_data):
    if np.any(train_data[row, 0:3] != [1, 6, 1]):
        target_train[row] = 0
    row += 1

target_test = np.ones(len(test_data))
row = 0
while row < len(test_data):
    if np.any(test_data[row, 0:3] != [1, 6, 1]):
        target_test[row] = 0
    row += 1

sample_train = train_data[:, [5,6,7,8,9,10,11,13,15,16,17,18,19,20,21]]
sample_test = test_data[:, [5,6,7,8,9,10,11,13,15,16,17,18,19,20,21]]

scaler = preprocessing.MinMaxScaler().fit(sample_train)
sample_train_scaled = scaler.transform(sample_train)
sample_test_scaled = scaler.transform(sample_test)

#clf = ExtraTreesClassifier(n_estimators=100)
clf = svm.SVC(kernel='linear')
clf.fit(sample_train, target_train)

prediction = clf.predict(sample_test)
print(metrics.confusion_matrix(target_test, prediction))
print(metrics.roc_auc_score(target_test, prediction))

clf = svm.OneClassSVM(kernel='linear', nu= 0.1)
clf.fit(sample_train_scaled)

print(clf.decision_function(sample_train_scaled))
print(clf.predict(sample_train_scaled))

y = np.array([9, 4, 5, -1, 6, -1, 5])
#y = clf.predict(sample_train_scaled)
index= np.where(y == -1)
print(index[0])
z = index[0]
y = np.array([x for x in y])
print(y)
index = np.where(y != -1)
print(index)
print(y[index[0]])

clf = EmpiricalCovariance()
clf.fit(sample_train_scaled)
print(clf.mahalanobis(sample_train_scaled))
"""