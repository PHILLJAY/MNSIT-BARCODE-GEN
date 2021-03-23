import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import barcode as bar
from PIL import Image as pil

# WORKING CODE SHOW

# from tensorflow.keras.datasets import mnist
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# sample = 2
# image = X_train[sample]

# fig = plt.figure()
# plt.imshow(image, cmap='gray')
# plt.show()

# loads image
im = pil.open("Herbert.jpg")
px = im.load()

# im.show()

# Test Code Down Here Ignore
# print(px[4, 4])
# print(px[6, 6])
# testMore = px[6, 6] + px[4, 4]
# print(testMore)

# print("height is " + str(height))
# print("width is " + str(width))
# print(list(im.getdata(0)))

width, height = im.size  # Creates variables for width and height from the image
pxList = list(im.getdata(0))  # Creates a list of each individual pixel using only the R channel
barcode = []  # Creates a list for the barcode itself

# Loops through each row
for x in range(width):
    # Debug code, ignore
    # print("checkpoint1")
    # print("current value of x is: " + str(x))

    temp = 0  # Temp int to hold the amount of white and black images
    for y in range(height):  # Cycles through each column
        # print((int(x) * int(width)) + y)
        # print("checkpoint2")

        # The value of the pixel ranges from 0-255, 255 being 100% white, 0 Being 100% black
        # If the pixel is under 128 the pixel is black (on average) and 1 is added to temp
        if pxList[(x * width) + y] < 128:
            temp += 1
        else:
            temp -= 1
    print(temp)  # Debug code
    # If temp is <= 1 then it is on average a white pixel so a 0 is appended to the barcode
    if temp <= 1:
        barcode.append(0)
    else:
        barcode.append(1)
# The same code as above but moving through columns first then rows second
for y in range(height):
    print("checkpoint1")
    print("current value of x is: " + str(y))
    temp = 0
    for x in range(width):
        print((int(x) * int(width)) + y)
        print("checkpoint2")
        if pxList[(x * width) + y] < 128:
            temp += 1
        else:
            temp -= 1
    print(temp)
    if temp <= 1:
        barcode.append(0)
    else:
        barcode.append(1)

print(barcode)
