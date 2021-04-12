
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


mnist = tf.keras.datasets.mnist

print("What position in the MNIST dataset?")

search = int(input())

(temp, labels), (imgtest2, labeltest) = mnist.load_data()
codes = []
imgtest = []
print("Creating barcodes for comparison set. 10000 total images")
codes.append(makebar(temp, search))
for x in range(0, len(imgtest2)):
    imgtest.append(makebar(imgtest2, x))

test = codes[0]
final = []

for x in range(0, len(imgtest2)):
    dis = ham(test, imgtest[x])
    if dis <= 5:
        final.append(x)

AI1 = tf.keras.models.load_model('num_confirm.model')
AI2 = tf.keras.models.load_model('num_confirm.model')
identify = AI1.predict([temp])
num = np.argmax(identify[search])
print("Here is the original image")
plt.imshow(temp[search])
plt.show()

print(len(final),  "Candidates Found")
candidates = AI2.predict([imgtest2])
for x in range(0, len(final)):
    candidates = AI2.predict([imgtest2])
    y = final[x]
    num2 = np.argmax(candidates[y])
    if num == num2:
        print("Here is a similar image")
        plt.imshow(imgtest2[y])
        plt.show()
        break


#print(codes[0])
#print(imgtest[0])
#print(ham(codes[0], imgtest[0]))
