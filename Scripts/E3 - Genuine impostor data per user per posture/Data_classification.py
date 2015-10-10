""" We will create a complete csv file containing genuine and impostor data
 that can be used for preliminary classification tasks.
 We will sample 150 impostor strokes for every user irrespective of the genuine strokes count.
 Thus we will sample 5 random impostor strokes from every impostor user"""

import numpy as np
import os
def GI(posture):
    i_user = 1
    i_impostor = 1
    block =1
    while block <= 6:
        while i_user <= 31:
            currentdirectory = os.getcwd()  # get the directory.
            parentdirectory = os.path.abspath(currentdirectory + "/../..")  # Get the parent directory(2 levels up)
            path = parentdirectory + '\Output Files\E3-Genuine Impostor data per user per posture/posture-'+str(posture)+'/User-'+str(i_user)+''
            if not os.path.exists(path):
                os.makedirs(path)

            genuine = np.genfromtxt('../../Output Files/E2-Genuine User-6Split/Posture-'+str(posture)+'/GenuineUser-'+str(i_user)+'/1-'+str(i_user)+'-'+str(posture)+'-'+str(block)+'.csv', dtype=float, delimiter=',')
            GI = genuine
            while i_impostor <= 31:
                if i_user != i_impostor:
                    impostor = np.genfromtxt('../../Output Files/E2-Data Split per User/Posture-'+str(posture)+'/1-'+str(i_impostor)+'-'+str(posture)+'.csv', dtype=float, delimiter=',')
                    random = np.random.randint(len(impostor), size=5)
                    randomimpostor = impostor[random, :]

                    GI = np.append(GI, randomimpostor, axis=0)
                i_impostor += 1
            np.savetxt('../../Output Files/E3-Genuine Impostor data per user per posture/posture-'+str(posture)+'/User-'+str(i_user)+'/1-'+str(i_user)+'-'+str(posture)+'-'+str(block)+'-GI.csv', GI, delimiter=',')
            i_impostor = 1
            i_user += 1
        i_user = 1
        i_impostor = 1
        block += 1


GI(3)

"""
find the length of genuine data of session x
lets say len = l
divide the length by 30 to get the number of strokes from each user
take a floor of that number

"""