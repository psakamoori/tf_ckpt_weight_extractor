import tensorflow as tf
import os
import sys

sess = tf.Session()

saver = tf.train.import_meta_graph('sample.meta')
res = saver.restore(sess, 'sample.ckpt')

all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

# Extracting weights and bias values
for i, v in enumerate(all_var):
    v_ = sess.run(v)
    print(v_)


