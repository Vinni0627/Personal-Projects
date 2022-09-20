import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model = tf.keras.models.load_model('HandwritingAlg.model')

testnumber = 1

while os.path.isfile(f"C:\\Users\\vpk12\\Personal Projects\\Python\\Handwriting Neural Net\\testcases\\test{testnumber}.png"):
    image = cv.imread(f"C:\\Users\\vpk12\\Personal Projects\\Python\\Handwriting Neural Net\\testcases\\test{testnumber}.png",0)
    image = np.invert(np.array([image]))
    prediction = model.predict(image)
    print(f'This digit is most likely a {np.argmax(prediction)}')
    plt.imshow(image[0], cmap=plt.cm.binary)
    plt.show()
    testnumber += 1
print ('end')