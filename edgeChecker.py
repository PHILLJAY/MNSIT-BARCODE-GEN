from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# returns the sum of the outside "frame"
def edge_check(num):
    image = X_train[num]
    big_temp = 0

    for y in range(2):              # checks the top 2 rows
        for i in range(28):
            big_temp += image[i, y]
    for y in range(2, 26):
        for i in range(2):          # checks the leftmost column
            big_temp += image[i, y]
        for i in range(26, 28):     # checks the rightmost column
            big_temp += image[i, y]
    for y in range(26, 28):
        for i in range(28):         # checks the bottom 2 rows
            big_temp += image[i, y]
    return big_temp                 # returns the sum of every pixel tested


temp = 0
for x in range(60000):              # checks each image in the dataset and adds it to a temp
    temp += edge_check(x)

average = (temp / 60000) / 216      # calculates the average value for a pixel found in the "frame"
print("the average value of the 2 edge pixels is: " + str(average))
