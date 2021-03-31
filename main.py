# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image as pil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image as pil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image as pil

# WORKING CODE SHOW

from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

sample = 256
image = X_train[sample]
print(image)


# diagonal barcode
def makebar(Images, p):
    barcode = []
    image = Images[p]
    for x in range(2, 26):
        temp = 0
        for y in range(2, 26):
            temp += image[x, y]
            if (temp / 24) < 48:
                barcode.append(0)
            else:
                barcode.append(1)
        for y in range(2, 26):
            temp = 0
            for x in range(2, 26):
                temp += image[x, y]
            if (temp / 24) < 48:
                barcode.append(0)
            else:
                barcode.append(1)
        for i in range(0, 12):
            temp = 0
            for x in range(i + 1):
                temp += image[(12 - i + x + 14), (x + 14)]
            testValue = temp / (i + 1)
            if testValue < 48.0:
                barcode.append(0)
            else:
                barcode.append(1)
        for i in range(0, 13):
            temp = 0
            for x in range(13-i):
                temp += image[x + 14 + i, x + 14]
            testValue = temp / (i + 1)
            if testValue < 48.0:
                barcode.append(0)
            else:
                barcode.append(1)
        return barcode


mnist = tf.keras.datasets.mnist

(temp, labels), (imgtest2, labeltest) = mnist.load_data()
codes = []
imgtest = []
for x in range(0, len(temp)):
    codes.append(makebar(temp, x))
for x in range(0, len(imgtest2)):
    imgtest.append(makebar(imgtest2, x))

codes = tf.keras.utils.normalize(codes, axis=1)
imgtest = tf.keras.utils.normalize(imgtest, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(codes, labels, epochs=5)

predictions = model.predict([imgtest])
print("I predict this image is a: ")
print(np.argmax(predictions[5]))

print(labeltest[5])
plt.imshow(imgtest2[5])
plt.show()