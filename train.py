import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing

numEpochs = 5
epoch_size = 1000000
batch = 1000
data = preprocessing.getTrainBatch(epoch_size)
train = data["train"][:int(epoch_size*.8)]
test = data["train"][int(epoch_size*.8):]
trainLabels = data["label"][:int(epoch_size*.8)]
testLabels = data["label"][int(epoch_size*.8):]
ids = data['id']
features = len(train[0])

X = tf.placeholder(tf.float32,[None,features],name='features')
Y = tf.placeholder(tf.int64,[None,5,10],name='labels')


W1 = tf.Variable(tf.truncated_normal([features, 128], mean=0, stddev=np.sqrt(6/(features+128))), name='weights1')
b1 = tf.Variable(tf.truncated_normal([128],mean=0, stddev=np.sqrt(6/(features+128))), name='biases1')

y1 = tf.nn.tanh((tf.matmul(X, W1)+b1), name='activationLayer1')

W2 = tf.Variable(tf.random_normal([128, 256],mean=0,stddev=np.sqrt(6/(128+256))),name='weights2')
b2 = tf.Variable(tf.random_normal([256],mean=0,stddev=np.sqrt(6/(128+256))),name='biases2')

y2 = tf.nn.tanh((tf.matmul(y1, W2)+b2), name='activationLayer2')

W3 = tf.Variable(tf.random_normal([256, 512],mean=0,stddev=np.sqrt(6/(256+512))),name='weights3')
b3 = tf.Variable(tf.random_normal([512],mean=0,stddev=np.sqrt(6/(256+512))),name='biases3')


y3 = tf.nn.tanh((tf.matmul(y2, W3)+b3), name='activationLayer3')

W4 = tf.Variable(tf.random_normal([512, 1024],mean=0,stddev=1/np.sqrt(6/(512+1024))),name='weights4')
b4 = tf.Variable(tf.random_normal([1024],mean=0,stddev=1/np.sqrt(6/(512+1024))),name='biases4')


y4 = tf.nn.tanh((tf.matmul(y3,W4)+b4),name='activationLayer4')

W5 = tf.Variable(tf.random_normal([1024, 512],mean=0,stddev=1/np.sqrt(6/(512+1024))),name='weights5')
b5 = tf.Variable(tf.random_normal([512],mean=0,stddev=1/np.sqrt(6/(512+1024))),name='biases5')

y5 = tf.nn.tanh((tf.matmul(y4,W5)+b5),name='activationLayer5')

W6 = tf.Variable(tf.random_normal([512, 256],mean=0,stddev=1/np.sqrt(6/(512+256))),name='weights6')
b6 = tf.Variable(tf.random_normal([256],mean=0,stddev=1/np.sqrt(6/(512+256))),name='biases6')

y6 = tf.nn.sigmoid((tf.matmul(y5,W6)+b6),name='activationLayer6')

W7 = tf.Variable(tf.random_normal([256, 50],mean=0,stddev=1/np.sqrt(6/(50+256))),name='weights7')
b7 = tf.Variable(tf.random_normal([50],mean=0,stddev=1/np.sqrt(6/(50+256))),name='biases7')

#output = tf.nn.softmax((tf.matmul(y6,W7)+b7),name='activationOutputLayer')

output = tf.nn.sigmoid((tf.matmul(y6,W7)+b7),name='activationOutputLayer')

logits = tf.reshape(output,(-1,5,10))


#test = tf.argmax(probs,axis=2)


#cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(output),reduction_indices=[1]))
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=logits)

#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
correct = tf.equal(tf.argmax(logits,axis=2),tf.argmax(Y,axis=2))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32),name="Accuracy")

tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    writer = tf.summary.FileWriter('modelData')
    writer.add_graph(sess.graph)
    summary_merge = tf.summary.merge_all()
    for x in range(numEpochs):
        for i in range(int(.8*epoch_size/batch)):   
            start = i*batch
            end = start+batch
            x_feed = train[start:end]
            y_feed = trainLabels[start:end]
            #y_feed = np.reshape(y_feed,(-1,5),'C')

            id_set = ids[start:end]



            sess.run(train_step,feed_dict={X:x_feed, Y:y_feed})
            acc, evaluate,losses = sess.run([accuracy,summary_merge,loss],feed_dict={X:x_feed,Y:y_feed})
            writer.add_summary(evaluate)
            
            print("batch:",i,"accuracy:",acc,"loss:",losses)



    pID = ids[0]
    x_feed = train[0:1]
    y_feed = trainLabels[0:1]

    print(x_feed,y_feed)

    tProb, sMax, aMax, labl, corr = sess.run([logits,tf.nn.softmax(logits),tf.argmax(logits,axis=2),tf.argmax(y_feed,axis=2),correct], feed_dict={X:x_feed,Y:y_feed})

    print('ID:',pID,"\noutput:",tProb,'\n',"softmax:",sMax,'\n',"argmax:",aMax,'\n',"expected:",labl,'\ncorrect:',corr)
    print("\n")


