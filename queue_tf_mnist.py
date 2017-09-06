
import tensorflow as tf
#import tensorflow.contrib.keras as keras
#from tensorflow.contrib.keras.python.keras.layers import Input, Conv2D, MaxPooling2D
#from tensorflow.contrib.keras.python.keras.layers.core import Dropout, Dense, Flatten
#from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pdb
from progress_bar import InitBar

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

NUM_CLASSES = 10
cwd = os.getcwd()
H, W = 28, 28
TRAIN_DATA_LIST_FILE = os.path.join(os.getcwd(), "mnist_list_train.txt")
TEST_DATA_LIST_FILE = os.path.join(os.getcwd(), "mnist_list_test.txt")

class DataLoader():

    def __init__(self, data_list_file, shuffle): # Do I need to add a coord here?
        self.image_list, self.label_list = self.read_data_list(data_list_file)
        self.image_list_tensor = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.label_list_tensor = tf.convert_to_tensor(self.label_list, dtype=tf.int32)
        self.queue = tf.train.slice_input_producer([self.image_list_tensor, self.label_list_tensor], shuffle=shuffle) # This is training data, so shuffle it.
        self.image, self.label = self.read_data_from_disk(self.queue)
        #self.coord = coord

    def read_data_list(self, filepath):
        f = open(filepath, 'r')
        filenames = []
        labels = []
        for line in f:
            fname, label = line.split(' ')
            filenames.append(fname)
            labels.append(int(label))
        return filenames, labels

    def read_data_from_disk(self, input_queue):
        label = input_queue[1]
        image_content = tf.read_file(input_queue[0])
        image = tf.image.decode_png(image_content, channels=1)
        image = tf.divide(image, 255)
        image.set_shape((H,W,1))
        return image, label

    def dequeue(self, batch_size):
        #pdb.set_trace()
        image_batch, label_batch = tf.train.batch([self.image, self.label], batch_size)
        return image_batch, label_batch


def create_data_list():
    print('Creating a training list:')
    f = open(TRAIN_DATA_LIST_FILE, 'w')
    for i in range(NUM_CLASSES):
        pathname = os.path.join(cwd, 'mnist_png', 'training', str(i))
        for fname in os.listdir(pathname):
            fullfname = os.path.join(pathname, fname)
            f.write(fullfname + ' {}\n'.format(i))
    f.close()

    print('Creating a test list:')
    f = open(TEST_DATA_LIST_FILE, 'w')
    for i in range(NUM_CLASSES):
        pathname = os.path.join(cwd, 'mnist_png', 'testing', str(i))
        for fname in os.listdir(pathname):
            fullfname = os.path.join(pathname, fname)
            f.write(fullfname + ' {}\n'.format(i))
    f.close()


def model(input, do_rate, reuse):
    # pdb.set_trace()

    with tf.name_scope('conv1'):
        with tf.variable_scope('conv1', reuse=reuse):
            W_conv1 = tf.get_variable('w', [3,3,1,32])
            b_conv1 = tf.get_variable('b', [32])
        conv1 = tf.nn.conv2d(input=input, filter=W_conv1, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(conv1 + b_conv1)

    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(value=relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.name_scope('conv2'):
        with tf.variable_scope('conv2', reuse=reuse):
            W_conv2 = tf.get_variable('w', [3,3,32,32])
            b_conv2 = tf.get_variable('b', [32])
        conv2 = tf.nn.conv2d(input=pool1, filter=W_conv2, strides=[1,1,1,1], padding='VALID')
        relu2 = tf.nn.relu(conv2 + b_conv2)

    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(value=relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.name_scope('dense1'):
        with tf.variable_scope('dense1', reuse=reuse):
            W_dense1 = tf.get_variable('w', [6*6*32,128])
            b_dense1 = tf.get_variable('b', 128)
        flat  = tf.reshape(pool2, [-1,6*6*32], name='reshape')
        dense1= tf.matmul(flat, W_dense1)
        relu3 = tf.nn.relu(dense1 + b_dense1)

    with tf.name_scope('dropout'):
        dropout = tf.nn.dropout(relu3, do_rate)

    with tf.name_scope('output'):
        with tf.variable_scope('output', reuse=reuse):
            W_out = tf.get_variable('w', [128,NUM_CLASSES])
            b_out = tf.get_variable('b', [NUM_CLASSES])
        output = tf.matmul(dropout, W_out) + b_out

    return output



def main():

    do_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')

    BATCH_SIZE = 32*8
    WIDTH, HEIGHT = 28, 28

    coord = tf.train.Coordinator()
    data_reader = DataLoader(data_list_file=TRAIN_DATA_LIST_FILE, shuffle=True)
    input_data, input_labels = data_reader.dequeue(BATCH_SIZE)
    output = model(input=input_data, do_rate=do_rate, reuse=None)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_labels, logits=output, name='loss'))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    accuracy = tf.cast(tf.equal(input_labels, tf.cast(tf.argmax(output, 1), tf.int32)), tf.float32)
    batch_acc = tf.reduce_mean(accuracy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graph', sess.graph)

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

    pbar = InitBar()
    nsteps = 2000 # Each step processes a batch of 256 samples
    for i in range(nsteps+1):
        pbar(100.0*float(i)/float(nsteps))
        l, acc, _ = sess.run([loss, batch_acc, train_op], {do_rate:0.5})
        #print("Iteration {}: Training batch loss = {}, Training batch accuracy = {}".format(i, l, acc))
        if (i%1000==0):
            saver.save(sess, './snapshots/saved_model', global_step=i)

    print('\n--------------------------\n')
    coord.request_stop()
    coord.join(threads)
    sess.close()

    coord = tf.train.Coordinator()
    data_reader = DataLoader(data_list_file=TEST_DATA_LIST_FILE, shuffle=False)
    input_data, input_labels = data_reader.dequeue(1)
    output = model(input=input_data, do_rate=do_rate, reuse=True)
    restore_var = tf.global_variables()

    accuracy = tf.cast(tf.equal(input_labels, tf.cast(tf.argmax(output, 1), tf.int32)), tf.float32)
    batch_acc = tf.reduce_mean(accuracy)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    loader = tf.train.Saver(var_list=restore_var)
    loader.restore(sess, './snapshots/saved_model-{}'.format(nsteps))

    total_acc = 0
    for i in range(10000):
        acc = sess.run([batch_acc], {do_rate:1.0})
        total_acc += acc[0]

    print('\n--------------------------\n')
    print("Total test accuracy = {}".format(float(total_acc) / float(10000)))

    print('\n--------------------------\n')
    coord.request_stop()
    coord.join(threads)

    sess.close()


    writer.close()
if __name__ == '__main__':
    main()




