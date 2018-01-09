import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2



# Defining our traning and test set 
TRAIN_DIR = '.\Project_data\Door'
TEST_DIR = '.\Test_data'
IMG_SIZE = 100
LR = 0.001           # learning rate
MODEL_NAME = 'door-vs window'

def create_label(image_name):
    """ Create an one -hot encoded vector from image name"""
    word_label = image_name.split('.')[0]
    if word_label == 'window':
        return np.array([1,0])
    elif word_label == 'door':
        return np.array([0,1])
    
#Re-sizing the image 100x100 and grayscale it
def create_train_data():
    training_data =[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

# split the dataset into two
def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), create_label(img_num)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

#if dataset is not created
train_data = create_train_data()
test_data = create_test_data()

#if already created the test dataset
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')
train = train_data[:-500]
test = train_data[-500:]

x_train = np.array([i[0]for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
y_train = [i[1]for i in train]
x_test = np.array([i[0]for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
y_test = [i[1]for i in test]

#Building the model
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convet, 1024, activation='relu')
convnet = fropout(convnet, 0.8)
convnet = fully_connected(convet, 2, activation='softmax')

convnet = regression(convnet, optimizer = 'adam', learning_rate=LR, loss='categorical_crossentropy', name='input')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input':x_train}, {'target':y_train}, n_epoch=10, 
          validation_set=({'input':x_test},{'target':y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

fig = plt.figure(figsize=(16, 12))
for num, data in enumerate(test_data[:16]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(4, 4, num +1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out)==1:
        str_label = 'door'
    else:
        str_label = 'window'
        
    y.imshow(orig, cmap='gray')
    plt.title(str_lable)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    plt.show()
    
