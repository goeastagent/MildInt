import pandas as pd
import numpy as np
import random


class DataGenerator:
    def __init__(self,dissimilarity, n_dimension, n_sample, n_ts_length,err_sig):
        self.err_sig = err_sig
        self.n_dimension = n_dimension
        self.n_sample = n_sample
        self.n_ts_length = n_ts_length
        self.dissimilarity = dissimilarity

        self.networkA, self.networkB = self.generate_diss_network2()

        self.networkA = self.networkA.reshape(self.n_dimension,self.n_dimension)
        self.networkB = self.networkB.reshape(self.n_dimension,self.n_dimension)
        
        self.modalityA = []
        self.modalityB = []
        
        labels = [0,1]*(n_sample/2)
        ids = range(n_sample)
        data = pd.DataFrame()
        data['ID'] = ids
        data['label'] = labels
        data.to_csv('data/synthetic/sigd' + str(dissimilarity) + 'labels.csv',sep=',',index=False)

    def generate_diss_network1(self):
        A = np.array([ random.uniform(-1,1) for i in range(self.n_dimension*self.n_dimension)])
        B = np.array([ random.uniform(-1,1) for i in range(self.n_dimension*self.n_dimension)])
        counter_network = []
    
        for i in range(self.n_dimension*self.n_dimension):
            if A[i] < B[i]:
                delta = B[i] - A[i]
            else:
                delta = A[i] - B[i]
            counter_network.append(A[i] + delta*self.dissimilarity)

        return A, np.array(counter_network)
    
    def generate_diss_network2(self):
        A = np.array([ random.randint(0,1) for i in range(self.n_dimension*self.n_dimension)])
        B = A - self.dissimilarity
        B = np.array(map(lambda x : -x if x<0 else x, B))
        return A,B
        
    def set_modality(self, modality_indices):
        self.n_modality = []
        self.initial_value = []
        for modality_index in modality_indices:
            self.modalityA.append(self.networkA[np.expand_dims(modality_index,1), modality_index])
            self.modalityB.append(self.networkB[np.expand_dims(modality_index,1), modality_index])
            self.n_modality.append(len(modality_index))
            self.initial_value.append(np.random.uniform(-1,1,len(modality_index)))
        
    def activation(self,A,x):
        temp = A.dot(x)
        #return self.linear_multiplication(temp)
        return self.sigmoid(temp)
        #return self.tanh(temp)
    
    def tanh(self, x):
        size = len(x)
        return np.tanh(x) + np.random.normal(0,self.err_sig,size)
    
    def sigmoid(self,x):
        size = len(x)
        return 1/(1+np.exp(-x)) + np.random.normal(0,self.err_sig,size)
                
    def linear_multiplication(self,x):
        size = len(x)
        return x + np.random.normal(0,self.err_sig,size)
    
    def generate_data(self):
        n = len(self.n_modality)
        for i in range(n):
            result = pd.DataFrame()
            labels = pd.DataFrame()
            col_names = np.concatenate([['ID'],['X' + str(index) for index in range(self.n_modality[i])]])
            
            for j in range(self.n_sample):
                if j < 500 and random.randint(0,1)== 1:
                    continue
                    
                initial_value = np.random.uniform(-1,1,self.n_modality[i])
                
                if j%2 == 0:
                    currentA = self.activation(self.modalityA[i],self.initial_value[i])
                    result = result.append(pd.DataFrame([np.concatenate([[j],currentA])]))
                    for t in range(self.n_ts_length-1):
                        currentA = self.activation(self.modalityA[i],currentA)
                        result = result.append(pd.DataFrame([np.concatenate([[j],currentA])]))
                else:
                    currentB = self.activation(self.modalityB[i],self.initial_value[i])
                    result = result.append(pd.DataFrame([np.concatenate([[j],currentB])]))
                    for t in range(self.n_ts_length-1):
                        currentB = self.activation(self.modalityB[i],currentB)
                        result = result.append(pd.DataFrame([np.concatenate([[j],currentB])]))

            result.columns = col_names
            result['ID'] = result['ID'].astype(int)
            
            result.to_csv('data/synthetic/sigd' + str(self.dissimilarity) + 'modality' + str(i) + '.csv',sep=',',index=False)

for i in np.arange(.1,1.1,.1):
    g = DataGenerator(i,10,1000,10,.2)

    index = [[0,1,2,3,4],[3,4,5,6],[6,7,8],[8,9]]
    g.set_modality(index)
    g.generate_data()
    
