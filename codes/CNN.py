
# coding: utf-8

# # Milestone Report Part 2: CNN
# 
# This is the code a regular CNN is trained and tested.

# In[1]:

import tensorflow as tf
import numpy as np

import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import cv2
import math

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8 # was 0.8


# In[2]:

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,64])), # was 32, 64
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,64,64])),
               #                                  64 features
               'W_conv3':tf.Variable(tf.random_normal([3,3,3,64,32])),
               #                                  64 features
               'W_conv4':tf.Variable(tf.random_normal([3,3,3,32,32])),
               'W_fc':tf.Variable(tf.random_normal([1024,1024])),
               'W_fc2':tf.Variable(tf.random_normal([1024,1024])),
               'W_fc3':tf.Variable(tf.random_normal([1024,1024])),# was 54080, 1024
               'out':tf.Variable(tf.random_normal([1024, n_classes]))} # was 1024

    biases = {'b_conv1':tf.Variable(tf.random_normal([64])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_conv3':tf.Variable(tf.random_normal([32])),
                'b_conv4':tf.Variable(tf.random_normal([32])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
              'b_fc2':tf.Variable(tf.random_normal([1024])),
              'b_fc3':tf.Variable(tf.random_normal([1024])),# was 16
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.leaky_relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.leaky_relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)
    
    conv3 = tf.nn.leaky_relu(conv3d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool3d(conv3)
    
    conv4 = tf.nn.leaky_relu(conv3d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = maxpool3d(conv4)
    print(conv4.get_shape)
    fc = tf.reshape(conv4,[-1, 1024])# was 54080
    fc = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc2'])+biases['b_fc2'])
    fc = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc3'])+biases['b_fc3'])
    #fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


# In the above you have to absolutely use leak_relu instead of relu, because negative input kill too many neuros if you use relu as you active function. leak_relu does not appear in tensorflow until the newest version (1.4.0), so update if you cannot run it.
# 
# Also I am totally overfitting here, you will see how learning error drop to 0 at the training stage. I think doing this in this case may do more good than harm given that how few the training samples I used. But if you want to re-run it in the full data set with all 1500 samples, you should definitely avoid overfitting. Just uncomment the dropout layer to prevent overfit (sort of)

# In[3]:

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


# In[4]:

#much_data = np.load('muchdata-50-50-20.npy')



#train_data = much_data[:-30]
#validation_data = much_data[-30:]
#train_data = much_data
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
much_data = np.load('../../cap2input/output/muchdata-50-50-20-b.npy')
train_data = [[item[0],item[1]] for item in much_data]
#test_data = train_data
test_data = np.load('../../cap2input/output/testdata-50-50-20-b.npy')
test_data = [[item[0],item[1],item[2]] for item in test_data]

P_list = []
proba_list = []
proba_list_2 = []

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=tf.to_int32(y) ))
    optimizer = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(cost)
    
    hm_epochs = 30
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass
                    #print(str(e))
            
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            #for i in xrange(10):
            #    val_data = chunkIt(validation_data,10)[i]
            #    print('Accuracy:',accuracy.eval({x:[i[0] for i in val_data], y:[i[1] for i in val_data]}))
            #print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            
            #print accuracy
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            predictions = tf.argmax(prediction,1)
            proba = tf.cast(prediction, 'float')
            proba_2 = tf.nn.softmax(logits=prediction)
                        
        print('Done. Finishing accuracy:')
        for j in xrange(30):
            te_data = chunkIt(test_data,30)[j]
            #print('Accuracy:',accuracy.eval({x:[i[0] for i in te_data], y:[i[1] for i in te_data]}))
            P_list.append(predictions.eval({x: [i[0] for i in te_data]}))
            proba_list.append(proba.eval({x: [i[0] for i in te_data]}))
            proba_list_2.append((proba_2.eval({x: [i[0] for i in te_data]})))
            
        #print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        print('fitment percent:',successful_runs/total_runs)

# Run this locally:
train_neural_network(x)


# Finishing accuarcy looks pretty good... Let us save it and submit!

# In[5]:

feature_test = [np.sum(img[0]) for img in test_data]


# In[6]:

labels = pd.read_csv('../../cap2input/stage2_sample_submission.csv', index_col=0)
#labels.cancer.value_counts()


# In[7]:

patient_list = [item[2] for item in test_data]


# In[8]:

ori_list_proba = [item for sublist in proba_list for item in sublist]

ori_list_proba_true = [[-sublist[1],-sublist[0]] if (sublist[0]+sublist[1]<0) else sublist for sublist in ori_list_proba]


proba_list_true = [sublist/sum(sublist) for sublist in ori_list_proba_true]

#proba_list_0 = [sublist[0] for sublist in proba_list_true]
proba_list_1 = [0.2 if sublist[1]<0.2 else 0.8 if sublist[1]>0.8 else sublist[1] for sublist in proba_list_true]

#ori_list_proba = [np.exp(item) for sublist in proba_list for item in sublist]


# In[9]:

#ori_list_proba_2 = [item for sublist in proba_list_2 for item in sublist]
#print ori_list_proba_2


# In[10]:

labels['proba'] = proba_list_1
labels['patient'] = patient_list


# In[11]:

pred = labels[['patient','proba']].reset_index(drop=True)
pred.columns = ['id','cancer']


# In[12]:

#pred[pred == 0] = 0.4
#pred[pred == 1] = 0.6


# In[13]:

pred


# In[15]:

pred.to_csv('../../cap2input/output/new_sixth_sub.csv', index=False)


# That is about it! The result is not good since we only have 300 samples (actually fewer than that, probably only 270 of them due to unlabelled data). The result constantly pass the brenchmark, but only in 1 of the 4 times it will get close to a bronze medal. On average, it ranked around 220 out of 394 teams

# In[ ]:



