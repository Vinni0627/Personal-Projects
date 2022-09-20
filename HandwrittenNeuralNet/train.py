# -*- coding: utf-8 -*-
#Building and training a handwriting neural network to be able to accurately guess which number is what from a handwritten picture
#using tensorflow as AI library
#


import tensorflow as tf

emnist = tf.keras.datasets.mnist.load_data()
(train1,train2), (test1,test2) = emnist # loads testing and training data into variables
train1 = tf.keras.utils.normalize(train1,axis=1) #normalizing the datasets to be a value between 0-1
test1 = tf.keras.utils.normalize(test1, axis=1)
model = tf.keras.Sequential() #creates the model
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # makes each picture into a giant 1D array(from a picture into a line)
model.add(tf.keras.layers.Dense(256,activation='relu')) #using Dense layer to show that each neuron is connected to the next linearly
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax')) #output layer (only 10 digits 0-9)
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy', metrics=['accuracy']) #compiler method
model.fit(train1,train2, epochs=5) # number of iterations
model.save('HandwritingAlg.model')

