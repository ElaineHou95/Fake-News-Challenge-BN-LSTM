import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from bnlstm import BNLSTMCell, orthogonal_initializer
import pickle
import pandas as pd
import dataset

class Databatch:
    def __init__(self,data,max_seqlen = 0):
#    def __init__(self,data):
        self.size=len(data)
        self.x_title=[]
        self.x_body=[]
        self.y=[]
        self.seqlen_title=[]
        self.seqlen_body=[]
        self.current_batch=0
        for sample in data:
            self.x_title.append(sample[0])
            self.x_body.append(sample[1])
            if len(sample) == 3:
                self.y.append(sample[2])
            if max_seqlen ==0:
                self.seqlen_title.append(len(sample[0]))
                self.seqlen_body.append(len(sample[1]))
            else:
                self.seqlen_title.append(max_seqlen)
                self.seqlen_body.append(max_seqlen)
        if max_seqlen == 0:
            max_seqlen=max(self.seqlen_body+self.seqlen_title)

        #padding the samples with zero vectors
        for i in range(len(self.x_title)):
            self.x_title[i]+=[[0]*50]*(max_seqlen-len(self.x_title[i]))
            self.x_body[i]+=[[0]*50]*(max_seqlen-len(self.x_body[i]))
    # return the next batch of the data from the data set.
    def next(self,batch_size):
        if self.current_batch+batch_size<self.size:
            self.current_batch+=batch_size
#            print(len(self.y))
            if len(self.y):
                return self.x_title[self.current_batch:self.current_batch+batch_size],\
                        self.x_body[self.current_batch:self.current_batch+batch_size],\
                        self.y[self.current_batch:self.current_batch+batch_size],\
                        self.seqlen_title[self.current_batch:self.current_batch+batch_size],\
                        self.seqlen_body[self.current_batch:self.current_batch+batch_size]
            else:
                return self.x_title[self.current_batch:self.current_batch+batch_size],\
                    self.x_body[self.current_batch:self.current_batch+batch_size],\
                    self.seqlen_title[self.current_batch:self.current_batch+batch_size],\
                    self.seqlen_body[self.current_batch:self.current_batch+batch_size]
        else:
            temp=self.current_batch
            self.current_batch=self.current_batch+batch_size-self.size
            batch_x_title=self.x_title[temp:]+self.x_title[:self.current_batch]
            batch_x_body=self.x_body[temp:]+self.x_body[:self.current_batch]
            batch_seqlen_title=self.seqlen_title[temp:]+self.seqlen_title[:self.current_batch]
            batch_seqlen_body=self.seqlen_body[temp:]+self.seqlen_body[:self.current_batch]

            if len(self.y):
                batch_y=self.y[temp:]+self.y[:self.current_batch]
                return batch_x_title,batch_x_body,batch_y,batch_seqlen_title,batch_seqlen_body
            else:
                return batch_x_title,batch_x_body,batch_seqlen_title,batch_seqlen_body
    # return the length of the longest sequence
    def max_seqlen(self):
        return max(self.seqlen_body+self.seqlen_title)


# loading the data
data=pickle.load(open("train_stances.p","rb"))
data2=pickle.load(open("test_stances.p","rb"))
data3=pickle.load(open("val_stances.p","rb"))
#print (data.shape)
size=len(data)
trainset=Databatch(data)
testset=Databatch(data2)
valset=Databatch(data3)


#
#network parameters
learning_rate=0.001
training_iters=100000
batch_size=128
display_step=10

#seq_max_len=max(trainset.max_seqlen(),testset.max_seqlen())

n_input=50
n_hidden=60
n_classes=4


x_title=tf.placeholder("float",[None,None,n_input])
x_body=tf.placeholder("float",[None,None,n_input])
y=tf.placeholder("float",[None,n_classes])
seqlen_title=tf.placeholder(tf.int32,[None])
seqlen_body=tf.placeholder(tf.int32,[None])
training = tf.placeholder(tf.bool)



weights={'out':tf.Variable(tf.truncated_normal([n_hidden,n_classes]))}
biases={'out':tf.Variable(tf.truncated_normal([1,n_classes]))}


def RNN(x_title,x_body,seqlen_title,seqlen_body,weights,biases):

#    lstm_cell=rnn.BasicLSTMCell(n_hidden)
    lstm_cell=BNLSTMCell(n_hidden, training)
    print("testing_1")
    outputs_title,states_title=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=x_title,sequence_length=seqlen_title,dtype=tf.float32)

    with tf.variable_scope('scope1',reuse=None):
        print("testing_2")
        outputs_body,states_body=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=x_body,sequence_length=seqlen_body,dtype=tf.float32)
        print("testing_3")
        temp1=tf.stack([tf.range(tf.shape(seqlen_title)[0]),seqlen_title-1],axis=1)
        temp2=tf.stack([tf.range(tf.shape(seqlen_body)[0]),seqlen_body-1],axis=1)
        return tf.matmul(tf.multiply(tf.gather_nd(outputs_title,temp1),tf.gather_nd(outputs_body,temp2)),weights['out'])+biases['out']


pred=tf.nn.softmax(RNN(x_title,x_body,seqlen_title,seqlen_body,weights,biases))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
#gvs = optimizer.compute_gradients(cross_entropy)
#capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#train_step = optimizer.apply_gradients(capped_gvs)
#pred=tf.nn.softmax(RNN(x_title,x_body,seqlen_title,seqlen_body,weights,biases))

#cost =tf.reduce_mean(cross_entropy)
#cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
#optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step=1
    count = 0
    test_dataset = dataset.DataSet("test_data.csv")
    head = []
    body_ID = []
    stance = []

    while step*batch_size<training_iters:
#    while step*batch_size<7000:
        print("step:",step)
        batch_x_title,batch_x_body,batch_y,batch_seqlen_title,batch_seqlen_body=trainset.next(batch_size)
        for i in range(len(batch_seqlen_title)):
            if batch_seqlen_title[i] == 0:
                batch_seqlen_title[i] = 1
        for i in range(len(batch_seqlen_body)):
            if batch_seqlen_body[i] == 0:
                batch_seqlen_body[i] = 1
        sess.run(optimizer,feed_dict={x_title:batch_x_title,x_body:batch_x_body,y:batch_y,seqlen_title:batch_seqlen_title,seqlen_body:batch_seqlen_body,training: True})
        loss=sess.run(cross_entropy,feed_dict={x_title:batch_x_title,x_body:batch_x_body,y:batch_y,seqlen_title:batch_seqlen_title,seqlen_body:batch_seqlen_body,training: False})
        print(loss)
        #sess.run(optimizer,feed_dict={x_title:batch_x_title,x_body:batch_x_body,y:batch_y,seqlen_title:batch_seqlen_title,seqlen_body:batch_seqlen_body,training: True})
        if step%display_step==0:
            acc=sess.run(accuracy,feed_dict={x_title:batch_x_title,x_body:batch_x_body,y:batch_y,seqlen_title:batch_seqlen_title,seqlen_body:batch_seqlen_body,training: False})
            #loss,_=sess.run([cross_entropy, train_step],feed_dict={x_title:batch_x_title,x_body:batch_x_body,y:batch_y,seqlen_title:batch_seqlen_title,seqlen_body:batch_seqlen_body,training: True})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step+=1
    print("Optimization Finished!")
    val_x_title=valset.x_title
    val_x_body=valset.x_body
    val_y=valset.y
    val_seqlen_title=valset.seqlen_title
    val_seqlen_body=valset.seqlen_body

    for i in range(len(val_seqlen_title)):
        if val_seqlen_title[i] == 0:
            val_seqlen_title[i] = 1
    for i in range(len(val_seqlen_body)):
        if val_seqlen_body[i] == 0:
            val_seqlen_body[i] = 1
    val_pred = sess.run(tf.argmax(pred,1),feed_dict={x_title:val_x_title,x_body:val_x_body,y:val_y,seqlen_title:val_seqlen_title,seqlen_body:val_seqlen_body,training: False})
    confusion = tf.confusion_matrix(labels=tf.argmax(valset.y,1), predictions=val_pred, num_classes=n_classes)
    print(sess.run(confusion))
    print("Test Accuracy:",sess.run(accuracy,feed_dict={x_title:val_x_title,x_body:val_x_body,y:val_y,seqlen_title:val_seqlen_title,seqlen_body:val_seqlen_body,training: False}))

    while count<testset.size:
        #generate result file
        test_x_title,test_x_body,test_seqlen_title,test_seqlen_body=testset.next(1)
        if test_seqlen_title == [0]:
            test_seqlen_title = [1]
        if test_seqlen_body == [0]:
            test_seqlen_body = [1]
        pred_label = sess.run(tf.argmax(pred,1),feed_dict={x_title:test_x_title,x_body:test_x_body,seqlen_title:test_seqlen_title,seqlen_body:test_seqlen_body,training: False})
        head.append(test_dataset.stances[count]['Headline'])
        body_ID.append(test_dataset.stances[count]['Body ID'])
        print(count)
        if pred_label == [0]:
            stance.append('discuss')
        elif pred_label == [1]:
            stance.append('agree')
        elif pred_label == [2]:
            stance.append('disagree')
        elif pred_label == [3]:
            stance.append('unrelated')
        count+=1
    print("Generate Finished!")
    dataframe = pd.DataFrame({'Headline':head,'Body ID':body_ID,'Stance':stance})
    dataframe.to_csv('answer.csv', index=False, encoding='utf-8')

