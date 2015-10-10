
import numpy as np
import os
import copy

def users_per_postures(posture):
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '\Output Files\E2-Data Split per User/Posture-'+str(posture)+''
    if not os.path.exists(path):
        os.makedirs(path)

    data = np.genfromtxt("../../Output Files/Device_Posture/1-"+str(posture)+".csv", dtype=float, delimiter=',')
    i_user = 1
    rowno = 0
    count = 0
    totallength = len(data)
    lengthofusers = []
    while i_user <= 31:
        while rowno < totallength:
            if np.all(data[rowno, 0:3] == [1, i_user, posture]):
                count += 1
            rowno += 1
        lengthofusers = np.append(lengthofusers, count)
        count = 0
        rowno = 0
        i_user += 1

    usercount = 0
    sum = 0
    while usercount <= 30:
        i_user = usercount + 1
        np.savetxt("../../Output Files/E2-Data Split per User/Posture-"+str(posture)+"/1-"+str(usercount+1)+"-"+str(posture)+".csv", data[sum: sum+lengthofusers[usercount]], delimiter=",")
        sum = sum + lengthofusers[usercount]
        usercount += 1

#users_per_postures(3)

# Split each user in 6 parts

def Genuine_Spit(number, posture):

    i_user = 1
    multiplier = 0
    while i_user <= 31:
        currentdirectory = os.getcwd()  # get the directory.
        parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
        path = parentdirectory + '\Output Files\E2-Genuine User-'+str(number)+'Split/Posture-'+str(posture)+'/GenuineUser-'+str(i_user)+''
        if not os.path.exists(path):
            os.makedirs(path)

        data = np.genfromtxt("../../Output Files/E2-Data Split per User/Posture-"+str(posture)+"/1-"+str(i_user)+"-"+str(posture)+".csv", dtype=float, delimiter=',')
        remainder = len(data) % number
        countpersplit = (len(data)-remainder)/number
        while multiplier < number:
            data_split = data[multiplier*countpersplit:(multiplier+1)*countpersplit, :]
            np.savetxt('../../Output Files/E2-Genuine User-'+str(number)+'Split/Posture-'+str(posture)+'/GenuineUser-'+str(i_user)+'/1-'+str(i_user)+'-'+str(posture)+'-'+str(multiplier+1)+'.csv', data_split, delimiter=',')
            multiplier += 1
        i_user += 1
        multiplier = 0

#Genuine_Spit(6, 3)

def Genuine_Session_Spit(posture):

    i_user = 1
    while i_user <= 31:
        currentdirectory = os.getcwd()  # get the directory.
        parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
        path = parentdirectory + '\Output Files\E2-Genuine User-Session Split/Posture-'+str(posture)+'/GenuineUser-'+str(i_user)+''
        if not os.path.exists(path):
            os.makedirs(path)

        data = np.genfromtxt("../../Output Files/E2-Data Split per User/Posture-"+str(posture)+"/1-"+str(i_user)+"-"+str(posture)+".csv", dtype=float, delimiter=',')
        sample = np.empty((0,22), float)
        session = 1
        totallength = len(data)
        row = 0
        while session <= 8:
            while row < totallength:
                if np.all(data[row, 3] == [session]):
                    stroke = copy.deepcopy(data[row, :])
                    sample = np.vstack((sample, stroke))
                row += 1
            np.savetxt('../../Output Files/E2-Genuine User-Session Split/Posture-'+str(posture)+'/GenuineUser-'+str(i_user)+'/1-'+str(i_user)+'-'+str(posture)+'-'+str(session)+'.csv', sample, delimiter=',')
            sample = np.empty((0,22), float)
            row = 0
            session += 1
        i_user += 1

Genuine_Session_Spit(3)

