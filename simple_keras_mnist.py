
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras.python.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.core import Dropout, Dense, Flatten
from tensorflow.contrib.keras.python.keras.models import Model
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
        for fname in os.listdir(os.path.join(cwd, 'mnist_png', 'training', str(i))):
            fullfname = os.path.join(cwd, 'mnist_png', 'training', str(i), fname)
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
        for fname in os.listdir(os.path.join(cwd, 'mnist_png', 'testing', str(i))):
            img = cv2.imread(os.path.join(cwd, 'mnist_png', 'testing', str(i), fname), 0)
            test_images.append(img)
            test_labels.append(i)
    test_images = np.stack(test_images, axis=0)
    test_labels = np.array(test_labels)
    print('test data has the shape of {}\n'.format(test_images.shape))

    return (training_images, training_labels, test_images, test_labels)


def main():
    training_images, training_labels, test_images, test_labels = load_dataset()
    
    # plt.imshow(training_images[:,:,0], cmap='gray')
    # plt.show()

    perm_train = np.random.permutation(training_labels.size)
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

    # Defiining the network
    input_layer = Input(shape=(HEIGHT, WIDTH, 1), name='input_layer')
    cnn1 = Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same', activation='relu')(input_layer)
    maxpool = MaxPooling2D(pool_size=2)(cnn1)
    cnn2 = Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid', activation='relu')(maxpool)
    maxpool = MaxPooling2D(pool_size=2)(cnn2)
    flat = Flatten()(maxpool)
    dense1 = Dense(units=128, activation='relu')(flat)
    dropout = Dropout(.5)(dense1)
    output_layer = Dense(units=NUM_CLASSES, activation='softmax')(dropout)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])

    # pdb.set_trace()

    print(model.summary())

    model.fit(x=training_images, y=training_labels, batch_size=BATCH_SIZE, epochs=30, verbose=1, validation_data=(test_images,test_labels))

    accuracy = model.evaluate(x=test_images, y=test_labels, batch_size=BATCH_SIZE)

    print('test score = {}'.format(accuracy))
    
if __name__ == '__main__':
    main()




