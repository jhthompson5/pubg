import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing

epoch_size = 50000
batch = 10000
data = preprocessing.getTrainBatch(epoch_size)
train = data["train"][:int(epoch_size*.8)]
test = data["train"][int(epoch_size*.8):]
trainLabels = data["label"][:int(epoch_size*.8)]
testLabels = data["label"][int(epoch_size*.8):]
ids = data['id']
features = len(train[0])

X = tf.placeholder(tf.float32,[None,features],name='features')
Y = tf.placeholder(tf.float32,[None,10000],name='labels')


W1 = tf.Variable(tf.truncated_normal([features, 128], mean=0, stddev=1 / np.sqrt(features)), name='weights1')
b1 = tf.Variable(tf.truncated_normal([128],mean=0, stddev=1 / np.sqrt(features)), name='biases1')

y1 = tf.nn.tanh((tf.matmul(X, W1)+b1), name='activationLayer1')

W2 = tf.Variable(tf.random_normal([128, 256],mean=0,stddev=1/np.sqrt(features)),name='weights2')
b2 = tf.Variable(tf.random_normal([256],mean=0,stddev=1/np.sqrt(features)),name='biases2')

y2 = tf.nn.tanh((tf.matmul(y1, W2)+b2), name='activationLayer2')

W3 = tf.Variable(tf.random_normal([256, 512],mean=0,stddev=1/np.sqrt(features)),name='weights3')
b3 = tf.Variable(tf.random_normal([512],mean=0,stddev=1/np.sqrt(features)),name='biases3')


y3 = tf.nn.tanh((tf.matmul(y2, W3)+b3), name='activationLayer3')

W4 = tf.Variable(tf.random_normal([512, 1024],mean=0,stddev=1/np.sqrt(features)),name='weights4')
b4 = tf.Variable(tf.random_normal([1024],mean=0,stddev=1/np.sqrt(features)),name='biases4')


y4 = tf.nn.tanh((tf.matmul(y3,W4)+b4),name='activationLayer4')

W5 = tf.Variable(tf.random_normal([1024, 2048],mean=0,stddev=1/np.sqrt(features)),name='weights5')
b5 = tf.Variable(tf.random_normal([2048],mean=0,stddev=1/np.sqrt(features)),name='biases5')

y5 = tf.nn.tanh((tf.matmul(y4,W5)+b5),name='activationLayer4')

W6 = tf.Variable(tf.random_normal([2048, 4096],mean=0,stddev=1/np.sqrt(features)),name='weights6')
b6 = tf.Variable(tf.random_normal([4096],mean=0,stddev=1/np.sqrt(features)),name='biases6')

y6 = tf.nn.sigmoid((tf.matmul(y5,W6)+b6),name='activationLayer4')

W7 = tf.Variable(tf.random_normal([4096, 10000],mean=0,stddev=1/np.sqrt(features)),name='weights7')
b7 = tf.Variable(tf.random_normal([10000],mean=0,stddev=1/np.sqrt(features)),name='biases7')

output = tf.nn.softmax((tf.matmul(y6,W7)+b7),name='activationOutputLayer')



#cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(output),reduction_indices=[1]))
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=output)
#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
correct = tf.equal(tf.argmax(output,axis=1),Y)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float16),name="Accuracy")

tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    writer = tf.summary.FileWriter('modelData')
    writer.add_graph(sess.graph)
    summary_merge = tf.summary.merge_all()
    for i in range(int(.8*epoch_size/batch)):   
        start = i*batch
        end = start+batch
        x_feed = train[start:end]
        y_feed = trainLabels[start:end]
        

        id_set = ids[start:end]

        sess.run(train_step,feed_dict={X:x_feed, Y:y_feed})


        acc, evaluate,losses = sess.run([accuracy,summary_merge,loss],feed_dict={X:x_feed,Y:y_feed})
        writer.add_summary(evaluate)
        
        print("batch:",i,"accuracy:",acc,"loss:",losses)
