import numpy as np
from numpy import *
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA

def reduced_dimension(posture):
    i_user = 1
    session = 1
    while i_user <= 31:
        currentdirectory = os.getcwd()  # get the directory.
        parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
        path = parentdirectory + '\Output Files\Reduced Dimensional Dataset/posture-'+str(posture)+'/GenuineUser'+str(i_user)+''
        if not os.path.exists(path):
            os.makedirs(path)

        while session <= 8:
            data = np.genfromtxt("../../Output Files/E2-Genuine User-Session Split/Posture-"+str(posture)+"/GenuineUser-"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(session)+".csv", dtype=float, delimiter=",")

            userinformation = data[:, [0,1,2,3,4]]
            sample_train = data[:, [5,6,7,8,9,10,11,13,15,16,17,18,19,20,21]]
            scaler = preprocessing.MinMaxScaler().fit(sample_train)
            sample_train_scaled = scaler.transform(sample_train)

            pca = PCA(n_components=7)
            sample_train_pca = pca.fit(sample_train_scaled).transform(sample_train_scaled)

            completedata = np.column_stack((userinformation, sample_train_pca))


            np.savetxt("../../Output Files/Reduced Dimensional Dataset/Posture-"+str(posture)+"/GenuineUser"+str(i_user)+"/1-"+str(i_user)+"-"+str(posture)+"-"+str(session)+".csv", completedata, delimiter=',')

            session += 1
        session = 1
        i_user += 1

reduced_dimension(3)
