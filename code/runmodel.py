import tensorflow as  tf
v1=tf.Variable(tf.constant(1.0,shape=[1]),name="1v1")
v2=tf.Variable(tf.constant(2.0,shape=[1]),name="2v2")
result=v1+v2
init_op=tf.global_variables_initializer()
saver=tf.train.Saver({"v1":v1,"v2":v2})
with tf.Session() as sess:
    saver.restore(sess,"../model/model.cpkt")
    print(sess.run(result))

# saver=tf.train.import_meta_graph("../model/model.cpkt.meta")
# with tf.Session() as sess:
#     saver.restore(sess,"../model/model.cpkt")
#     print( sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))