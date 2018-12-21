from mmrnn import *

batch_size = 32

m1_hidden = 2
m2_hidden = 3
m3_hidden = 4


dm = DataManager()

m1, y_m1, IDs_m1, seqlen_m1 = dm.generate_m1()
m2, y_m2, IDs_m2, seqlen_m2 = dm.generate_m2()
m3, y_m3, IDs_m3, seqlen_m3 = dm.generate_m3()

overlapIDs = dm.generate_overlapIDs([IDs_m1, IDs_m2, IDs_m3])
m = MMRNN()

m.append_component('m1', m1.shape[2], m1_hidden, m1.shape[1])
m.append_component('m2', m2.shape[2], m2_hidden, m2.shape[1])
m.append_component('m3', m3.shape[2], m3_hidden, m3.shape[1])

m.append_data('m1', IDs_m1, y_m1, seqlen_m1)
m.append_data('m2', IDs_m2, y_m2, seqlen_m2)
m.append_data('m3', IDs_m3, y_m3, seqlen_m3)

# 5-fold CV
test_folds = dm.generate_crossvalidation_set(overlapIDs)
accuracy = []
for i in range(5):
    m.append_test_overlapIDs(test_folds[i])
    trainIDs = overlapIDs[~overlapIDs.isin(test_folds[i])]
    m.append_training_overlapIDs(trainIDs)

    with tf.varialbe_scope('fold run'):
        m.build_integrative_network()
        m.training(batch_size)
        
    accuracy.append(m.evaluate_accuracy())
    tf.reset_default_graph()
    
print(np.mean(accuracy))
