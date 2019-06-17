import tensorflow as tf
from sklearn.model_selection import train_test_split
import vecData
from RegressionModel import RegressionModel
#跑这个就可以了
INPUT_FEATURE=300

OUTPUT_NUM=2
BATCH_SIZE=110
LAYER1_NODE=1000

x=tf.placeholder(tf.float32,[None,INPUT_FEATURE])
y=tf.placeholder(tf.float32,[None,OUTPUT_NUM])
#定义w和b
weights1=tf.Variable(tf.truncated_normal([INPUT_FEATURE,LAYER1_NODE],stddev=0.1))
biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NUM],stddev=0.1))
biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NUM]))

l1=tf.nn.relu(tf.matmul(x,weights1)+biases1)
pre=tf.nn.relu(tf.matmul(l1,weights2)+biases2)

loss_money=tf.reduce_mean(abs(y[:,0]-pre[:,0]))
loss_xingqi=tf.reduce_mean(abs(y[:,1]-pre[:,1]))
loss_total=loss_money+loss_xingqi
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss_total)
r=RegressionModel()

#标注数据_使用特征
# X,Y=r.get_data()
#
# X_train1, X_test1, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=23)
# X_test = X_test1[:, 1:6]#5-10
# X_train = X_train1[:, 1:6]
# y_train = X_train1[:, -4:-2]
# y_test = X_test1[:, -4:-2]

#使用文本转型向量向量
X,Y=vecData.get_vec_data('11')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=23)
#
dataset_size=len(X_train)

#Tensorflow跑NN
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     STEPS=10000
#     for i in range(STEPS):
#         start=(i*BATCH_SIZE)%dataset_size
#         end=min(start+BATCH_SIZE,dataset_size)
#         sess.run(train,feed_dict={x:X_train[start:end],y:y_train[start:end]})
#         if i%1000==0 or i==9999:
#            loss_m=sess.run(loss_money,feed_dict={x:X_train,y:y_train})
#            loss_x = sess.run(loss_xingqi, feed_dict={x: X_train, y: y_train})
#            loss_t = sess.run(loss_total, feed_dict={x: X_train, y: y_train})
#            print("--------------")
#            print("Train:After %d training step(s)，loss money is %g,"
#                  "loss xinqi is %g,total loss is %g"%(i,loss_m,loss_x,loss_t))
#
#            loss_m=sess.run(loss_money,feed_dict={x:X_test,y:y_test})
#            loss_x = sess.run(loss_xingqi, feed_dict={x: X_test, y: y_test})
#            loss_t = sess.run(loss_total, feed_dict={x: X_test, y: y_test})
#            print("Test:After %d training step(s)，loss money is %g,"
#                  "loss xinqi is %g,total loss is %g"%(i,loss_m,loss_x,loss_t))

#svm等模型 需要调用那个可以去RegressionModel里面看
a=RegressionModel()
a.SVR_model_xq(X_train,X_test,y_train,y_test)
