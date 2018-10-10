import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
import random
import pickle
from math import floor

def getTrainBatch(batchSize):
    filename = 'data/train.csv'
    row_count = len(open(filename,'rb').readlines())
    skip = sorted(random.sample(range(row_count),(row_count-batchSize)))
    if 0 in skip:
        skip.remove(0)
    df = pd.read_csv(
         filename,
         header=0, 
         skiprows=skip
    )
    keepHeads = list(df.columns)
    keepHeads.remove("winPlacePerc")
    
    trainHeads = keepHeads
    trainHeads.remove("Id")
    trainHeads.remove("groupId")
    trainHeads.remove("matchId")

    training = {}
    for i in trainHeads:
        training[i] = list(df[i])
        ''' j = [list(df[i])]
        scaler = Normalizer().fit(j)
        j_scaled = scaler.transform(j)
        training[i] = j_scaled[0,:]'''

    #Get each row of training together as entry to list of lists    

    labels = []
    for i in list(df["winPlacePerc"]):
        bfr = [[0 for x in range(10)] for k in range(5)]
        dec = i
        for x in range(5):
            bit = int(floor(dec))
            bfr[x][bit] = 1
            dec = round(dec-bit, 4-x)
            dec = dec * 10
        labels.append(bfr)


    trainSet = []
    for i in range(len(training['kills'])):
        bfr = []
        for key in training.keys():
            #if i == 0:
              #  print(key,":",training[key][i])
            bfr.append(training[key][i])
        trainSet.append(bfr)
    data = {
        'id': np.array(df["Id"]),
        'train': np.array(trainSet),
        'label': np.array(labels)
    }
    
    return data




