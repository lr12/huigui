import tensorflow as tf
v=tf.Variable(0,dtype=tf.float32,name="v")
ema=tf.train.ExponentialMovingAverage(0.99)
# saver=tf.train.Saver({"v/ExponentialMovingAverage":v})
saver=tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,"../model/model.ckpt")
    print(sess.run(v))