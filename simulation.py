
from mmrnn import *
import SimulationDataManager

import numpy as np
import tensorflow as tf
from random import shuffle


def get_baseline_single(activation,diss):
    db = SimulationDataManager.DataManager()
    data = pd.DataFrame()

    X,y,ids,seqlen = db.generate_modality('data/synthetic/' + activation + 'd' + str(diss) + 'modality' + str(0) + '.csv',1)
    X = pd.DataFrame(np.squeeze(X,axis=1))
    X.columns = [ 'X' + str(i) for i in range(X.shape[1])]

    data['ID'] = ids
    data['y'] = y
    data = pd.concat([data,X],axis=1)
    return data



def get_baseline_concate(activation,diss):
    db = SimulationDataManager.DataManager()
    data = pd.DataFrame()

    feature_cnt = 0
    X,y,ids,seqlen = db.generate_modality_by_tpoint('data/synthetic/' + activation + 'd' + str(diss) + 'modality' + str(0) + '.csv',1)
    X = pd.DataFrame(np.squeeze(X,axis=1))
    X.columns = [ 'X' + str(i) for i in range(feature_cnt, feature_cnt + X.shape[1])]
    feature_cnt += X.shape[1]

    data['ID'] = ids
    data['y'] = y
    data = pd.concat([data,X],axis=1)
    for i in range(1,4):
        temp =pd.DataFrame()
        X,y,ids,seqlen = db.generate_modality('data/synthetic/' + activation + 'd' + str(diss) + 'modality' + str(i) + '.csv',1)
        
        X = pd.DataFrame(np.squeeze(X,axis=1))
        X.columns = [ 'X' + str(i) for i in range(feature_cnt, feature_cnt + X.shape[1])]
        feature_cnt += X.shape[1]

        temp['ID'] =ids
        temp['y'] = y
        temp = pd.concat([temp,X],axis=1)

        data = pd.merge(data,temp,on=['ID','y'])

    data = data.dropna()
    return data
    
def baseline_single_function(activation,diss,f):
    data = get_baseline_single(activation,diss)

    acc = []
    for i in range(10):
        chunk = data.shape[0]//5
        data = data.sample(frac=1)
        for j in range(5):
            test = data[j*chunk:(j+1)*chunk]
            train = data[~data['ID'].isin(test['ID'])]

            train_X = train.drop(['ID','y'],axis=1)
            train_y = train['y']

            test_X = test.drop(['ID','y'],axis=1)
            test_y = test['y']

            clf = f()
            clf.fit(train_X,train_y)
            estimated_y = clf.predict(test_X)

            acc.append(sum(estimated_y==test_y)/float(len(test_y)))

    return np.mean(acc)

def baseline_multimodal_function(activation,diss,f):
    data = get_baseline_concate(activation,diss)

    acc = []
    for i in range(10):
        chunk = data.shape[0]//5
        data = data.sample(frac=1)
        for j in range(5):
            test = data[j*chunk:(j+1)*chunk]
            train = data[~data['ID'].isin(test['ID'])]

            train_X = train.drop(['ID','y'],axis=1)
            train_y = train['y']

            test_X = test.drop(['ID','y'],axis=1)
            test_y = test['y']

            clf = f()
            clf.fit(train_X,train_y)
            estimated_y = clf.predict(test_X)

            acc.append(sum(estimated_y==test_y)/float(len(test_y)))

    return np.mean(acc)

    
def singlemodality_function(activation,diss,k,ts):
    db = SimulationDataManager.DataManager()
    IDs = []
    data_X = []
    data_y = []
    data_seqlen = []
    data_ID = []
        
    for i in range(4):
        X, y, ids, seqlen = db.generate_modality('data/synthetic/' + activation + 'd' + str(diss) + 'modality' + str(i) + '.csv',ts)
        data_X.append(X)
        data_y.append(y)
        data_seqlen.append(seqlen)
        data_ID.append(ids)
        IDs.append(ids)
    
    overlapIDs = db.get_overlapIDs(IDs)
    accuracy = []
    for i in range(10):
        chunk_size = len(overlapIDs)//5
        for j in range(5):
            m = MMRNN()
            
            m.append_component('modality' + str(k), data_X[k].shape[2], data_X[k].shape[2]+1, data_X[k].shape[1])
            m.append_data('modality' + str(k),data_ID[k], data_X[k], data_y[k], data_seqlen[k])

            testIDs = overlapIDs[j*chunk_size:(j+1)*chunk_size]
            trainIDs = overlapIDs[~overlapIDs.isin(testIDs)]
            m.append_test_overlapIDs(testIDs)
            m.append_training_overlapIDs(trainIDs)
            with tf.variable_scope('single_run'):
                m.build_integrative_network()
                m.training(32)
            accuracy.append(m.evaluate_balanced_accuracy())
            tf.reset_default_graph()
            
    return np.array(accuracy).mean()

def multimodality_function(activation,diss,ts):
    db = SimulationDataManager.DataManager()
    IDs = []
    data_X = []
    data_y = []
    data_seqlen = []
    data_ID = []
        
    for i in range(4):
        X, y, ids, seqlen = db.generate_modality('data/synthetic/' + activation + 'd' + str(diss) + 'modality' + str(i) + '.csv',ts)
        data_X.append(X)
        data_y.append(y)
        data_seqlen.append(seqlen)
        data_ID.append(ids)
        IDs.append(ids)
    
    overlapIDs = db.get_overlapIDs(IDs)
    accuracy = []
    for i in range(10):
        chunk_size = len(overlapIDs)//5
        for j in range(5):
            m = MMRNN()

            for k in range(4):
                m.append_component('modality' + str(k), data_X[k].shape[2], data_X[k].shape[2]+1, data_X[k].shape[1])
                m.append_data('modality' + str(k),data_ID[k], data_X[k], data_y[k], data_seqlen[k])

            testIDs = overlapIDs[j*chunk_size:(j+1)*chunk_size]
            trainIDs = overlapIDs[~overlapIDs.isin(testIDs)]
            m.append_test_overlapIDs(testIDs)
            m.append_training_overlapIDs(trainIDs)
            with tf.variable_scope('single_run'):
                m.build_integrative_network()
                m.training(32)
            accuracy.append(m.evaluate_balanced_accuracy())
            tf.reset_default_graph()
            
    return np.array(accuracy).mean()

def baseline_single_proposed(activation,diss):
    db = SimulationDataManager.DataManager()
    IDs = []
    data_X = []
    data_y = []
    data_seqlen = []
    data_ID = []
        
    ts = 1
    for i in range(1):
        X, y, ids, seqlen = db.generate_modality('data/synthetic/' + activation + 'd' + str(diss) + 'modality' + str(i) + '.csv',ts)
        data_X.append(X)
        data_y.append(y)
        data_seqlen.append(seqlen)
        data_ID.append(ids)
        IDs.append(ids)
    
    overlapIDs = db.get_overlapIDs(IDs)
            
    accuracy = []
    for i in range(10):
        chunk_size = len(overlapIDs)//5
        for j in range(5):
            m = MMRNN()
            
            for k in range(1):
                m.append_component('modality' + str(k), data_X[k].shape[2], data_X[k].shape[2]+1, data_X[k].shape[1])
                m.append_data('modality' + str(k),data_ID[k], data_X[k], data_y[k], data_seqlen[k])
    
            testIDs = overlapIDs[j*chunk_size:(j+1)*chunk_size]
            trainIDs = overlapIDs[~overlapIDs.isin(testIDs)]
            m.append_test_overlapIDs(testIDs)
            m.append_training_overlapIDs(trainIDs)
            with tf.variable_scope('single_run'):
                m.build_integrative_network()
                m.training(32)
            accuracy.append(m.evaluate_balanced_accuracy())
            tf.reset_default_graph()
            
    return np.array(accuracy).mean()

def baseline_multimodal_proposed(activation,diss):
    db = SimulationDataManager.DataManager()
    IDs = []
    data_X = []
    data_y = []
    data_seqlen = []
    data_ID = []
        
    ts = 1
    for i in range(4):
        X, y, ids, seqlen = db.generate_modality('data/synthetic/' + activation + 'd' + str(diss) + 'modality' + str(i) + '.csv',ts)
        data_X.append(X)
        data_y.append(y)
        data_seqlen.append(seqlen)
        data_ID.append(ids)
        IDs.append(ids)
    
    overlapIDs = db.get_overlapIDs(IDs)
            
    accuracy = []
    for i in range(10):
        chunk_size = len(overlapIDs)//5
        for j in range(5):
            m = MMRNN()
            
            for k in range(4):
                m.append_component('modality' + str(k), data_X[k].shape[2], data_X[k].shape[2], data_X[k].shape[1])
                m.append_data('modality' + str(k),data_ID[k], data_X[k], data_y[k], data_seqlen[k])
    
            testIDs = overlapIDs[j*chunk_size:(j+1)*chunk_size]
            trainIDs = overlapIDs[~overlapIDs.isin(testIDs)]
            m.append_test_overlapIDs(testIDs)
            m.append_training_overlapIDs(trainIDs)
            with tf.variable_scope('single_run'):
                m.build_integrative_network()
                m.training(32)
            accuracy.append(m.evaluate_balanced_accuracy())
            tf.reset_default_graph()
            
    return np.array(accuracy).mean()


def baseline_multimodal_study():
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    activations = ['lin','sig','tanh']
    activation = 'sig'
    
    dissimilarities = np.arange(0.1,1.1,.1)

    for dissimilarity in dissimilarities:
        print('----------' + str(dissimilarity) + '-------------')
        a = baseline_multimodal_function(activation, dissimilarity, svm.SVC)
        b = baseline_multimodal_function(activation, dissimilarity, RandomForestClassifier)
        c = baseline_multimodal_function(activation, dissimilarity, LogisticRegression)
        d = baseline_multimodal_proposed(activation, dissimilarity)
        print(str(a) + ',' + str(b) + ','+ str(c) + ',' + str(d))
    

def baseline_single_study():
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    activations = ['lin','sig','tanh']
    activation = 'sig'
    
    dissimilarities = np.arange(0.1,1.1,.1)

    for dissimilarity in dissimilarities:
        print('----------' + str(dissimilarity) + '-------------')
        a = baseline_single_function(activation, dissimilarity, svm.SVC)
        b = baseline_single_function(activation, dissimilarity, RandomForestClassifier)
        c = baseline_single_function(activation, dissimilarity, LogisticRegression)
        d = baseline_single_proposed(activation, dissimilarity)
        print(str(a) + ',' + str(b) + ','+ str(c) + ',' + str(d))


import sys
t = int(sys.argv[1])

def longitudinal_multimodality_study():
    dissimilarities = np.arange(0.4,.8,.1)
    activations = ['','sig','tanh']
    activation='sig'
    
    for dissimilarity in dissimilarities:
        # dissimilarity = 1.0
        print(multimodality_function(activation,dissimilarity,t))
        # break
    
    
def singlemodality_study():
    dissimilarities = np.arange(0.1,1.1,.1)
    activations = ['lin','sig','tanh']
    activation='sig'
    k = 0
    ts = 10
    k=1
    for dissimilarity in dissimilarities:
        dissimilarity = 1.0
        s = []
        for t in range(1,11):
            s.append(singlemodality_function(activation,dissimilarity,k,t))
        print(s)


#singlemodality_study()
longitudinal_multimodality_study()
#baseline_single_study()
#baseline_multimodal_study()

