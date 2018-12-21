import pandas as pd
import numpy as np

from random import shuffle
from datetime import datetime

class DataManager(object):
    def __init__(self):
        self.labels = pd.read_csv('data/label.csv',sep=',')

    def generate_overlapRIDs(self, RIDs_list):
        sets = iter(map(set, RIDs_list))
        result = sets.next()
        for s in sets:
            result = result.intersection(s)
        return pd.Series(list(result))

    def generate_training_test_RIDs(self, RIDs):
        test_folds = []
        y = self.get_labels(RIDs)

        positive_RIDs = np.array(RIDs[y == 1])
        negative_RIDs = np.array(RIDs[y == 0])
        
        shuffle(positive_RIDs)
        shuffle(negative_RIDs)
        
        chunks_positive = len(positive_RIDs)//5
        chunks_negative = len(negative_RIDs)//5

        for i in range(5):
            test_folds.append(np.concatenate((positive_RIDs[i*chunks_positive:(i+1)*chunks_positive], negative_RIDs[i*chunks_negative :(i+1)*chunks_negative])))

        return test_folds

    def get_labels(self,RIDs):
        labels = self.labels
        y = []
        for RID in RIDs:
            y.append(labels[labels['RID'] == RID].iloc[0]['Y'])
        return np.array(y)
    
    def generate_MRI(self):
        MRI = pd.read_csv('data/MRI.csv',sep=',')
        RIDs = MRI['RID']

        # Scaling MRI
        temp = MRI[['BL_ICV','BL_HippVol','BL_Thick_EntCtx']]
        MRI = (temp - temp.mean())/(temp.max() - temp.min())
        MRI['RID'] = RIDs
    
        MRI = np.expand_dims(MRI.drop('RID',axis=1),axis=1)
        y = self.get_labels(RIDs)
        seqlen = np.array(np.repeat(1,len(RIDs)))
        
        return MRI,y, RIDs,seqlen
        
    def generate_demo(self):
        demo = pd.read_csv('data/demographic_information.csv',sep=',')
    
        # Scaling 
        demo['PTGENDER'] = [ 1.0 if i=='Male' else 0.0 for i in demo['PTGENDER']]
        data = demo[['PTGENDER','AGE','PTEDUCAT','APOE4']]

        RIDs = demo['RID']
        demo = (data - data.mean())/(data.max() - data.min())
        demo['RID'] = RIDs

        demo = np.expand_dims(demo.drop('RID',axis=1),axis=1)
        y = self.get_labels(RIDs)
        seqlen = np.array(np.repeat(1,len(RIDs)))
        
        return demo,y, RIDs, seqlen

    def generate_cog(self,max_seq_len):
        cog = pd.read_csv('data/cognitive_performance.csv',sep=',')

        RIDs = cog['RID']
        # scaling
        temp = cog[['ADNI_MEM','ADNI_EF']]
        temp = (temp - temp.mean())/(temp.max() - temp.min())
        temp['RID'] = RIDs
        cog = temp

        X = []
        seqlen = []
        
        RIDs = RIDs.unique()
        
        for RID in RIDs:
            record = cog[cog['RID'] == RID]
            if max_seq_len < len(record):
                seq_length = max_seq_len
            else: 
                seq_length = len(record) 
            seqlen.append(seq_length)
        
            fixed_vector = []
            for i in range(max_seq_len):
                if i < seq_length:
                    temp = np.array(record.iloc[i,:][['ADNI_MEM','ADNI_EF']].tolist())
                    temp[np.isnan(temp)] = 0
                else:
                    temp = [.0 for j in range(2)]
                fixed_vector.append(temp)
            X.append(fixed_vector)
            
        X = np.array(X)
        seqlen = np.array(seqlen)
        Y = self.get_labels(RIDs)
        return X,Y,RIDs,seqlen
    
    def generate_csf(self,max_seq_len):
        csf = pd.read_csv('data/cerebrospinal_fluid.csv',sep=',')
    
        RIDs = csf['RID']
        # scaling
        temp = csf[['LOGABETA','LOGTAU','LOGPTAU','LOGPTAU/ABETA','LOGTAU/ABETA']]
        temp = (temp-temp.mean())/(temp.max()-temp.min())
        temp['RID'] = RIDs
        csf = temp

        X = []
        seqlen = []
        RIDs = RIDs.unique()

        for RID in RIDs:            
            record = csf[csf['RID'] == RID]
            if max_seq_len < len(record):
                seq_length = max_seq_len
            else:
                seq_length = len(record)
            seqlen.append(seq_length)
            
            fixed_vector = []
            for i in range(max_seq_len):
                if i < seq_length:
                    temp = record.iloc[i,:][['LOGABETA','LOGTAU','LOGPTAU','LOGPTAU/ABETA','LOGTAU/ABETA']].tolist()
                    if np.isnan(temp).any():
                        temp = [.0 for j in range(5)]
                else:
                    temp = [.0 for j in range(5)]
                fixed_vector.append(temp)
            X.append(fixed_vector)

        X = np.array(X)
        seqlen = np.array(seqlen)
        Y = self.get_labels(RIDs)
        return X,Y,RIDs,seqlen
