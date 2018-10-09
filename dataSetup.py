import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import Normalizer

filename = 'data/train.csv'
df = pd.read_csv(
        filename,
        header=0, 
)
keepHeads = list(df.columns)
keepHeads.remove("winPlacePerc")

trainHeads = keepHeads
trainHeads.remove("Id")
trainHeads.remove("groupId")
trainHeads.remove("matchId")

training = {}
for i in trainHeads:
    j = [list(df[i])]
    scaler = Normalizer().fit(j)
    j_scaled = scaler.transform(j)
    training[i] = j_scaled[0,:]

#Get each row of training together as entry to list of lists    

labels = []
for i in list(df["winPlacePerc"]):
    index = int(i*10000)
    bfr = [0]*10001
    bfr[index] = 1
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

pickle.dump(data,open('data/processed.dat','wb'),-1)