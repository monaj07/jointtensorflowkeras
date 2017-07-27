
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NUM_CLASSES = 10
cwd = os.getcwd()

def load_dataset():
    # load training data
    training_images = []
    training_labels = []
    print('Loading training data:')
    for i in range(NUM_CLASSES):
        print(i)
        pathname = os.path.join(cwd, 'mnist_png', 'training', str(i))
        for fname in os.listdir(pathname):
            fullfname = os.path.join(pathname, fname)
            img = cv2.imread(fullfname, 0)
            training_images.append(img)
            training_labels.append(i)
    training_images = np.stack(training_images, axis=0)
    training_labels = np.array(training_labels)
    print('training data has the shape of {}\n'.format(training_images.shape))

    # load test data
    test_images = []
    test_labels = []
    print('Loading test data:')
    for i in range(NUM_CLASSES):
        print(i)
        pathname = os.path.join(cwd, 'mnist_png', 'testing', str(i))
        for fname in os.listdir(pathname):
            img = cv2.imread(os.path.join(pathname, fname), 0)
            test_images.append(img)
            test_labels.append(i)
    test_images = np.stack(test_images, axis=0)
    test_labels = np.array(test_labels)
    print('test data has the shape of {}\n'.format(test_images.shape))

    return (training_images, training_labels, test_images, test_labels)

def get_w(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)

def get_b(shape, name):
    return tf.Variable(tf.zeros(shape), name=name)

def get_batch(step, bs, images, labels):
    n = images.shape[0]
    if (step+1)*bs > n:
        return images[step*bs:, :, :, :], labels[step*bs:, :]
    else:
        #pdb.set_trace()
        return images[step*bs:(step+1)*bs, :, :, :], labels[step*bs:(step+1)*bs, :]

def main():
    training_images, training_labels, test_images, test_labels = load_dataset()

    # plt.imshow(training_images[:,:,0], cmap='gray')
    # plt.show()

    N = training_labels.size
    Nt = test_labels.size
    perm_train = np.random.permutation(N)
    training_labels = training_labels[perm_train]
    training_images = training_images[perm_train, :, :] / 255.0
    training_images = np.expand_dims(training_images, -1)
    print(training_images.shape)
    test_images = test_images / 255.0
    test_images = np.expand_dims(test_images, -1)

    # pdb.set_trace()

    training_labels = to_categorical(training_labels, NUM_CLASSES)
    test_labels = to_categorical(test_labels, NUM_CLASSES)

    BATCH_SIZE = 32*8
    WIDTH, HEIGHT = 28, 28
    epochs = 30

    # Defiining the placeholders
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 1], name='data')
    input_labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name='labels')
    do_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')
    # pdb.set_trace()

    conv1 = tf.nn.conv2d(input=input_data, filter=get_w([3,3,1,32], 'conv1_w'), strides=[1,1,1,1], padding='SAME', name='conv1')
    relu1 = tf.nn.relu(conv1 + get_b([32], 'conv1_b'))
    pool1 = tf.nn.max_pool(value=relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
    conv2 = tf.nn.conv2d(input=pool1, filter=get_w([3,3,32,32], 'conv2_w'), strides=[1,1,1,1], padding='VALID', name='conv2')
    relu2 = tf.nn.relu(conv2 + get_b([32], 'conv2_b'))
    pool2 = tf.nn.max_pool(value=relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
    flat  = tf.reshape(pool2, [-1,6*6*32], name='reshape')
    dense1= tf.matmul(flat, get_w([6*6*32,128], 'dense1_w'), name='dense1')
    relu3 = tf.nn.relu(dense1 + get_b([128], 'dense1_b'), name='relu1')
    dropout1 = tf.nn.dropout(relu3, do_rate, name='dropout1')
    output = tf.matmul(dropout1, get_w([128, NUM_CLASSES], 'out_w'), name='output') + get_b([NUM_CLASSES], 'out_b')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=output, name='loss'))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    accuracy = tf.cast(tf.equal(tf.argmax(input_labels, 1), tf.argmax(output, 1)), tf.float32)

    # Training:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graph', sess.graph)
    
    for i in range(epochs):
        steps = (int)(np.ceil(float(N)/float(BATCH_SIZE)))
        print(steps)
        total_l = 0
        total_acc = 0
        for step in range(steps):
            x_in, y_in = get_batch(step, BATCH_SIZE, training_images, training_labels)
            l, acc, _ = sess.run([loss, accuracy, train_op], {input_data:x_in, input_labels:y_in, do_rate:0.5})
            total_l += l
            total_acc += np.sum(acc)
            #pdb.set_trace()
        total_acc /= np.float32(N)
        print("Epoch {}: Training loss = {}, Training accuracy = {}".format(i,total_l,total_acc))

    # Test:
    steps = (int)(np.ceil(float(Nt)/float(BATCH_SIZE)))
    for step in range(steps):
        x_in, y_in = get_batch(step, BATCH_SIZE, test_images, test_labels)
        acc, _ = sess.run([accuracy, train_op], {input_data:x_in, input_labels:y_in, do_rate:1})
        total_acc += np.sum(acc)
    total_acc /= np.float32(Nt)
    print('\n--------------------------\n')
    print("Test accuracy = {}".format(total_acc))
    sess.close()
    writer.close()
if __name__ == '__main__':
    main()




