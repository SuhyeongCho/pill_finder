import tensorflow as tf
import numpy as np
import cv2

def char_test(image):
    graph = tf.Graph()
    
    size = 48
    n_class = 37
    learning_rate = 0.001

    with graph.as_default():
        X = tf.placeholder(np.float32,shape=[None,size,size,1])
        Y = tf.placeholder(np.float32,shape=[None,n_class])
        keep_prob = tf.placeholder(np.float32)

        conv1 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=[2,2])
        conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=[2,2])
        conv3 = tf.layers.conv2d(inputs=pool2,filters=128,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=[2,2])

        pool_flat = tf.reshape(pool3,[-1,size*size*2])

        fc1 = tf.layers.dense(inputs=pool_flat,units=1024,activation=tf.nn.relu)
        fc1_drop = tf.layers.dropout(fc1,rate=keep_prob)
        fc2 = tf.layers.dense(inputs=fc1_drop,units=256,activation=tf.nn.relu)
        fc2_drop = tf.layers.dropout(fc2,rate=keep_prob)
        logits = tf.layers.dense(inputs=fc2_drop,units=n_class)


    with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            # tf.reset_default_graph()
            saver = tf.train.Saver()
            # saver = tf.train.import_meta_graph('char_saver/char-9900.meta')
            saver.restore(sess,'char_saver/char-10000')
            image = cv2.resize(image,(size,size))
            image = np.reshape(image,[-1,size,size,1])
            pred = sess.run(logits,feed_dict={X:image,keep_prob:1.0})
                
    return np.argmax(pred,1)
                