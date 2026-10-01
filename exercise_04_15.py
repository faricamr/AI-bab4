# Exercise 4.15 
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

model = VGG19(weights='imagenet')

n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters : ", s[3])

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Batasi channel yang ditampilkan jadi 8
n_filters, ix = 4, 1
n_channels = 8   # ← tambahan
pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(n_channels):   # ← ganti s[0] jadi n_channels
        ax = pyplot.subplot(n_filters, n_channels, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
pyplot.show()