
import numpy as np
from numpy import *
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

def infogain(posture):

    i_user = 1
    block = 1
    i = 0

    while block <= 6:
        currentdirectory = os.getcwd()  # get the directory.
        parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
        path = parentdirectory + '\Output Files\E4-Information Gain/posture-'+str(posture)+'/block-'+str(block)+''
        if not os.path.exists(path):
            os.makedirs(path)

        #gain = np.arange(527, dtype=float32).reshape(527, 1)
        gain = []
        while i_user <= 31:
            data = np.genfromtxt('../../Output Files/E3-Genuine Impostor data per user per posture/posture-'+str(posture)+'/user-'+str(i_user)+'/1-'+str(i_user)+'-'+str(posture)+'-'+str(block)+'-GI.csv', dtype=float, delimiter=',')

            target_train = np.ones(len(data))
            row = 0
            while row < len(data):
                if np.any(data[row, 0:3] != [1, i_user, posture]):
                    target_train[row] = 0
                row += 1

            sample_train = data[:, 3:]
            scaler = preprocessing.StandardScaler().fit(sample_train)
            sample_train_scaled = scaler.transform(sample_train)

            clf = ExtraTreesClassifier(n_estimators=100)
            clf.fit(sample_train_scaled, target_train)

            #print(clf.feature_importances_)
            gain = np.append(gain, clf.feature_importances_)
            #gain[i, 0] = clf.feature_importances_
            i_user += 1
        i_user = 1
        gain_metrix = np.transpose(gain.reshape(31, 17))
        np.savetxt("../../Output Files/E4-Information Gain/posture-"+str(posture)+"/block-"+str(block)+"/gain-"+str(block)+"-"+str(posture)+".csv", gain_metrix, delimiter=",")
        block += 1

infogain(1)