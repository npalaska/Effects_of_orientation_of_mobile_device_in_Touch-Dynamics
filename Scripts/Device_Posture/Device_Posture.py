""" This file will create separate csv files for every combination of device and posture.
However we maintain the device constant for our experiment"""
import numpy as np
import os

currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '\Output Files\Device_posture'
if not os.path.exists(path):
    os.makedirs(path)

def makefile(filename, posture):
    file = np.genfromtxt(filename, dtype=float, delimiter=',')
    i_user = 1
    device = 1
    lengthofusers = []
    rowno = 0
    count = 0
    totallength = len(file)
    """
    while i_user <= 31:
        while rowno < totallength:
            if np.all(file[rowno, 0:3] == [device, i_user, posture]):
                count += 1
            rowno += 1
        lengthofusers = np.append(lengthofusers, count)
        count = 0
        rowno = 0
        i_user += 1
    """

    count = 0
    while rowno < totallength:
        if np.all(file[rowno, [0,2]] == [device, posture]):
            count += 1
            print(file[rowno, [0,2]])
        if count == 1:
            start = rowno
        rowno += 1
    truncatedfile = file[start:count+start, :]


    np.savetxt('../../Output Files/Device_Posture/1-1.csv', truncatedfile, delimiter=',')

#makefile("../../Raw Data/H.csv", 1)
#makefile("../../Raw Data/H.csv", 2)
makefile("../../Raw Data/50 Features/H.csv", 1)