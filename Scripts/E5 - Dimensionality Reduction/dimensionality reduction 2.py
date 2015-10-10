import numpy as np
from numpy import *
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
from sklearn.lda import LDA
from sklearn import metrics


def All_Features(posture, trainblock):
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '\Output Files\E5-Dimensionality Reduction/posture-'+str(posture)+'/TrainBlock-'+str(trainblock)+''
    if not os.path.exists(path):
        os.makedirs(path)
    i_user = 1
    block = 1
    AUC = []
    while i_user <= 31:
        while block <= 6:
            train_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-"+str(posture)+"/User-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(trainblock)+"-GI.csv", dtype=float, delimiter=",")
            test_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-"+str(posture)+"/User-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(block)+"-GI.csv", dtype=float, delimiter=",")

            target_train = np.ones(len(train_data))
            row = 0
            while row < len(train_data):
                if np.any(train_data[row, 0:3] != [1, i_user, posture]):
                    target_train[row] = 0
                row += 1

            row = 0
            target_test = np.ones(len(test_data))
            while row < len(test_data):
                if np.any(test_data[row, 0:3] != [1, i_user, posture]):
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
            AUC.append(auc)

            block += 1

        block = 1
        i_user += 1
    print(AUC)
    AUC = np.array(AUC)
    AUC = AUC.reshape(31, 6)
    np.savetxt("../../Output Files/E5-Dimensionality Reduction/posture-"+str(posture)+"/TrainBlock-"+str(trainblock)+"/Performance with all features.csv", AUC, delimiter=",")

# All_Features(2, 1)

def PCA_reduction(posture, trainblock, componenet):
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '\Output Files\E5-Dimensionality Reduction/posture-'+str(posture)+'/TrainBlock-'+str(trainblock)+''
    if not os.path.exists(path):
        os.makedirs(path)
    i_user = 1
    block = 1
    AUC = []
    while i_user <= 31:
        while block <= 6:
            train_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-"+str(posture)+"/User-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(trainblock)+"-GI.csv", dtype=float, delimiter=",")
            test_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-"+str(posture)+"/User-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(block)+"-GI.csv", dtype=float, delimiter=",")

            target_train = np.ones(len(train_data))
            row = 0
            while row < len(train_data):
                if np.any(train_data[row, 0:3] != [1, i_user, posture]):
                    target_train[row] = 0
                row += 1

            row = 0
            target_test = np.ones(len(test_data))
            while row < len(test_data):
                if np.any(test_data[row, 0:3] != [1, i_user, posture]):
                    target_test[row] = 0
                row += 1

            sample_train = train_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17]]
            sample_test = test_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17]]
            scaler = preprocessing.MinMaxScaler().fit(sample_train)
            sample_train_scaled = scaler.transform(sample_train)
            sample_test_scaled = scaler.transform(sample_test)

            pca = PCA(n_components=componenet)
            sample_train_pca = pca.fit(sample_train_scaled).transform(sample_train_scaled)
            sample_test_pca = pca.transform(sample_test_scaled)

            clf = ExtraTreesClassifier(n_estimators=100)
            clf.fit(sample_train_pca, target_train)

            prediction = clf.predict(sample_test_pca)
            auc = metrics.roc_auc_score(target_test, prediction)
            AUC.append(auc)

            block += 1

        block = 1
        i_user += 1
    print(AUC)
    AUC = np.array(AUC)
    AUC = AUC.reshape(31, 6)
    np.savetxt("../../Output Files/E5-Dimensionality Reduction/posture-"+str(posture)+"/TrainBlock-"+str(trainblock)+"/PCA-"+str(componenet)+"-Component.csv", AUC, delimiter=",")

#PCA_reduction(2, 1, 9)

def LDA_reduction(posture, trainblock, componenet):
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '\Output Files\E5-Dimensionality Reduction/posture-'+str(posture)+'/TrainBlock-'+str(trainblock)+''
    if not os.path.exists(path):
        os.makedirs(path)
    i_user = 1
    block = 1
    AUC = []
    while i_user <= 31:
        while block <= 6:
            train_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-"+str(posture)+"/User-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(trainblock)+"-GI.csv", dtype=float, delimiter=",")
            test_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-"+str(posture)+"/User-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(block)+"-GI.csv", dtype=float, delimiter=",")

            target_train = np.ones(len(train_data))
            row = 0
            while row < len(train_data):
                if np.any(train_data[row, 0:3] != [1, i_user, posture]):
                    target_train[row] = 0
                row += 1

            row = 0
            target_test = np.ones(len(test_data))
            while row < len(test_data):
                if np.any(test_data[row, 0:3] != [1, i_user, posture]):
                    target_test[row] = 0
                row += 1

            sample_train = train_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17]]
            sample_test = test_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17]]
            scaler = preprocessing.MinMaxScaler().fit(sample_train)
            sample_train_scaled = scaler.transform(sample_train)
            sample_test_scaled = scaler.transform(sample_test)

            lda = LDA(n_components=componenet)
            sample_train_lda = lda.fit(sample_train_scaled, target_train).transform(sample_train_scaled)
            sample_test_lda = lda.transform(sample_test_scaled)

            clf = ExtraTreesClassifier(n_estimators=100)
            clf.fit(sample_train_lda, target_train)

            prediction = clf.predict(sample_test_lda)
            auc = metrics.roc_auc_score(target_test, prediction)
            AUC.append(auc)

            block += 1

        block = 1
        i_user += 1
    print(AUC)
    AUC = np.array(AUC)
    AUC = AUC.reshape(31, 6)
    np.savetxt("../../Output Files/E5-Dimensionality Reduction/posture-"+str(posture)+"/TrainBlock-"+str(trainblock)+"/LDA-"+str(componenet)+"-Component.csv", AUC, delimiter=",")

# LDA_reduction(1, 1, 7)

def ICA_reduction(posture, trainblock, componenet):
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '\Output Files\E5-Dimensionality Reduction/posture-'+str(posture)+'/TrainBlock-'+str(trainblock)+''
    if not os.path.exists(path):
        os.makedirs(path)
    i_user = 1
    block = 1
    AUC = []
    while i_user <= 31:
        while block <= 6:
            train_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-"+str(posture)+"/User-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(trainblock)+"-GI.csv", dtype=float, delimiter=",")
            test_data = np.genfromtxt("../../Output Files/E3-Genuine Impostor data per user per posture/posture-"+str(posture)+"/User-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(block)+"-GI.csv", dtype=float, delimiter=",")

            target_train = np.ones(len(train_data))
            row = 0
            while row < len(train_data):
                if np.any(train_data[row, 0:3] != [1, i_user, posture]):
                    target_train[row] = 0
                row += 1

            row = 0
            target_test = np.ones(len(test_data))
            while row < len(test_data):
                if np.any(test_data[row, 0:3] != [1, i_user, posture]):
                    target_test[row] = 0
                row += 1

            sample_train = train_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17]]
            sample_test = test_data[:, [3,4,5,6,7,9,11,12,13,14,15,16,17]]
            scaler = preprocessing.MinMaxScaler().fit(sample_train)
            sample_train_scaled = scaler.transform(sample_train)
            sample_test_scaled = scaler.transform(sample_test)

            ica = FastICA(n_components=componenet, max_iter=150)
            sample_train_ica = ica.fit(sample_train_scaled).transform(sample_train_scaled)
            sample_test_ica = ica.transform(sample_test_scaled)

            clf = ExtraTreesClassifier(n_estimators=100)
            clf.fit(sample_train_ica, target_train)

            prediction = clf.predict(sample_test_ica)
            auc = metrics.roc_auc_score(target_test, prediction)
            AUC.append(auc)

            block += 1

        block = 1
        i_user += 1
    print(AUC)
    AUC = np.array(AUC)
    AUC = AUC.reshape(31, 6)
    np.savetxt("../../Output Files/E5-Dimensionality Reduction/posture-"+str(posture)+"/TrainBlock-"+str(trainblock)+"/ICA-"+str(componenet)+"-Component.csv", AUC, delimiter=",")

ICA_reduction(1, 1, 9)