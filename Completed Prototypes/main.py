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

def ham(in1, in2):
    diff = 0
    for x in range (0, len(in1)):
        if in1 [x] != in2 [x]:
            diff+=1
    return diff
# diagonal barcode
def makebar(Images, p):
    threshhold = 48
    barcode = []
    image = Images[p]
    for x in range(2, 26):
        temp = 0
        for y in range(2, 26):
            temp += image[x, y]
            if (temp / 24) < threshhold:
                barcode.append(0)
            else:
                barcode.append(1)
        for y in range(2, 26):
            temp = 0
            for x in range(2, 26):
                temp += image[x, y]
            if (temp / 24) < threshhold:
                barcode.append(0)
            else:
                barcode.append(1)
        for i in range(0, 12):
            temp = 0
            for x in range(i + 1):
                temp += image[(12 - i + x + 14), (x + 14)]
            testValue = temp / (i + 1)
            if testValue < threshhold:
                barcode.append(0)
            else:
                barcode.append(1)
        for i in range(0, 13):
            temp = 0
            for x in range(13-i):
                temp += image[x + 14 + i, x + 14]
            testValue = temp / (i + 1)
            if testValue < threshhold:
                barcode.append(0)
            else:
                barcode.append(1)
        return barcode


def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0

    while low <= high:

        mid = (high + low) // 2

        # If x is greater, ignore left half
        if arr[mid] < x:
            low = mid + 1

        # If x is smaller, ignore right half
        elif arr[mid] > x:
            high = mid - 1

        # means x is present at mid
        else:
            return mid

    # If we reach here, then the element was not present
    return -1


mnist = tf.keras.datasets.mnist

print("What position?")

search = int(input())

(temp, labels), (imgtest2, labeltest) = mnist.load_data()
print(len(temp))
codes = []
imgtest = []
for x in range(0, len(temp)):
    codes.append(makebar(temp, x))
for x in range(0, len(imgtest2)):
    imgtest.append(makebar(imgtest2, x))

test = codes[search]
final = []
for x in range(0, len(imgtest2)):
    dis = ham(test, imgtest[x])
    if dis <= 5:
        final.append(x)

AI1 = tf.keras.models.load_model('num_confirm.model')
AI2 = tf.keras.models.load_model('num_confirm.model')
identify = AI1.predict([temp])
num = np.argmax(identify[search])
print(num)
plt.imshow(temp[search])
plt.show()

print(len(final))
candidates = AI2.predict([imgtest2])
for x in range(0, len(final)):
    candidates = AI2.predict([imgtest2])
    y = final[x]
    num2 = np.argmax(candidates[y])
    if num == num2:
        plt.imshow(temp[search])
        plt.show()
        print(num)
        plt.imshow(imgtest2[y])
        plt.show()
        print(num2)


#print(codes[0])
#print(imgtest[0])
#print(ham(codes[0], imgtest[0]))
