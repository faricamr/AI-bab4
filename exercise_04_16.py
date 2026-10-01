# Exercise 4.16
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

# load the model
model = VGG19(weights='imagenet')

# retrieve weights from the first hidden layer
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape

print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot ALL filters
n_filters, ix = s[3], 1

pyplot.figure(figsize=(15, 30))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
pyplot.show()