import pandas as pd
import numpy as np

from random import shuffle
from datetime import datetime


class DataManager(object):
    def __init__(self):
        pass

    def get_labels(self,filename, IDs):
        labels = pd.read_csv(filename,sep=',')
        y = []
        for ID in IDs:
            y.append(labels[labels['ID'] == ID].iloc[0]['label'])
        return np.array(y)

    def get_overlapIDs(self,IDs):
        overlapIDs = pd.Series(IDs[0])
        for ID in IDs:
            overlapIDs = overlapIDs[overlapIDs.isin(ID)]
        return pd.Series(overlapIDs)

    def remove_random_ts(X,seqlen):
        pass

    def remove_random_sample(ID,X,y,seqlen,num):
        remove_element = random.sample(range(500),num)

        index = ID.isin(remove_lement)
        ID = ID[~index]
        X = X[~index]
        y = y[~index]
        seqlen = seqlen[~index]

        return ID,X,y,seqlen
        

    def generate_modality_by_tpoint(self,filename,point):
        data = pd.read_csv(filename, sep=',')

        import re
        label_filename = re.sub('modality[0-9]','labels',filename)
        
        IDs = data['ID'].unique()
        dim = data.shape[1]
        
        X = []
        seqlen = []            
        
        for ID in IDs:
            record = data[data['ID'] == ID]
            record = record.drop('ID',axis=1)
            seqlen.append(1)

            fixed_vector = []
            X.append([record.iloc[point,:].tolist()])
            
        X=np.array(X)
        seqlen = np.array(seqlen)
        y = self.get_labels(label_filename,IDs)
        
        return X,y,IDs,seqlen
        
        
    def generate_modality(self,filename,max_seq_len):
        data = pd.read_csv(filename, sep=',')

        import re
        label_filename = re.sub('modality[0-9]','labels',filename)
        
        IDs = data['ID'].unique()
        dim = data.shape[1]
        
        X = []
        seqlen = []            
        
        for ID in IDs:
            record = data[data['ID'] == ID]
            record = record.drop('ID',axis=1)
            if max_seq_len < len(record):
                seq_length = max_seq_len
            else:
                seq_length = len(record)
            seqlen.append(seq_length)

            fixed_vector = []
            for i in range(max_seq_len):
                if i < seq_length:
                    temp = np.array(record.iloc[i,:].tolist())
                else:
                    temp = [0 for j in range(dim)]
                fixed_vector.append(temp)
            X.append(fixed_vector)
        X=np.array(X)
        seqlen = np.array(seqlen)
        y = self.get_labels(label_filename,IDs)
        
        return X,y,IDs,seqlen
            
                
