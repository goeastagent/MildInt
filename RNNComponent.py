# Jan 26th 2018


__author__ = 'Garam Lee'
__email__ = 'goeastagent@gmail.com'

import tensorflow as tf

#LSTMCell = tf.contrib.rnn.BasicLSTMCell
GRUCell = tf.contrib.rnn.GRUCell


class RNNComponent:
    def __init__(self, n_hidden, max_seq_len, x, y, seqlen, cell=None, optimizer=None):
        if cell == None:
            self.cell = GRUCell(n_hidden)
        else :
            self.cell = cell
            
        if optimizer == None:
            optimizer = tf.train.AdamOptimizer()

        self.x = x
        self.y = y
        self.seqlen = seqlen

        self.weights = tf.Variable(tf.random_normal([n_hidden, 1]))
        self.biases = tf.Variable(tf.random_normal([1]))
    
        x = tf.unstack(x, axis=1)

        self.outputs, self.states = tf.contrib.rnn.static_rnn(self.cell, x, dtype=tf.float32, sequence_length=seqlen)
        self.outputs = tf.stack(self.outputs)
        self.outputs = tf.transpose(self.outputs,[1,0,2])

        batch_size = tf.shape(self.outputs)[0]

        index = tf.range(0,batch_size) * max_seq_len + (seqlen - 1)

        self.outputs = tf.gather( tf.reshape(self.outputs, [-1, n_hidden]),index)
        self.pred = tf.nn.softmax(tf.matmul(self.outputs,self.weights) + self.biases)

        y = tf.expand_dims(self.y,axis=1)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=y))
        self.train = optimizer.minimize(self.loss)
        
    def training(self,sess,data,label,seqlen,batch_size):
        sample_size = len(seqlen)
        nb_iters = sample_size//batch_size

        for i in range(nb_iters):
            batch_x = data[i*batch_size:(i+1)*batch_size]
            batch_y = label[i*batch_size:(i+1)*batch_size]
            batch_seqlen = seqlen[i*batch_size:(i+1)*batch_size]
            feed_dict = { self.x : batch_x,
                          self.y : batch_y,
                          self.seqlen : batch_seqlen}
            loss, _ = sess.run([self.loss, self.train],feed_dict=feed_dict)
            
        if nb_iters % batch_size != 0:
            i+=1
            batch_x = data[i*batch_size:]
            batch_y = label[i*batch_size:]
            batch_seqlen = seqlen[i*batch_size:]
            
            feed_dict = { self.x : batch_x,
                          self.y : batch_y,
                          self.seqlen : batch_seqlen}
            
            loss, _ = sess.run([self.loss, self.train],feed_dict=feed_dict)
            
    def extract_output(self,sess,x,seqlen):
        feed_dict = {self.x : x, self.seqlen: seqlen}
        return sess.run(self.outputs,feed_dict=feed_dict)
