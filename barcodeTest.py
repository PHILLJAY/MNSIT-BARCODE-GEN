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

barcode = []
# horizontal barcode
# for x in range(2, 26):
#     temp = 0
#     for y in range(2, 26):
#         temp += image[x, y]
#     if (temp / 24) < 48:
#         barcode.append(0)
#     else:
#         barcode.append(1)
#
# # vertical barcode
# for y in range(2, 26):
#     temp = 0
#     for x in range(2, 26):
#         temp += image[x, y]
#     if (temp / 24) < 48:
#         barcode.append(0)
#     else:
#         barcode.append(1)

# diagonal barcode

for i in range(0, 12):
    temp = 0
    for x in range(i + 1):
        print("Coordinate [" + str(x + 14) + ", " + str(12 - i + x + 14) + "] | " + str(
            image[(12 - i + x + 14), (x + 14)]))
        temp += image[(12 - i + x + 14), (x + 14)]
    print("checkpoint " + str(i + 1) + " temp value is " + str(temp))
    print("Normalized temp value is " + str(temp / i))
    testValue = temp / (i + 1)
    if testValue < 48.0:
        barcode.append(0)
        print("appending 0")
    else:
        barcode.append(1)
        print("appending 1")

for i in range(0, 13):
    print(i)
    temp = 0
    for x in range(13-i):
        print("Coordinate [" + str(x + 14 + i) + ", " + str(x + 14) + "] | " + str(
             image[x + 14 + i, x + 14]))
        temp += image[x + 14 + i, x + 14]
    print("checkpoint " + str(i + 1) + " temp value is " + str(temp))
    print("Normalized temp value is " + str(temp / (13-i)))
    testValue = temp / (i + 1)
    if testValue < 48.0:
        barcode.append(0)
        print("appending 0")
    else:
        barcode.append(1)
        print("appending 1")

print(barcode)
checkThis = []

