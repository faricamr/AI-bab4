# Exercise 4.17
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

# ---------- Fungsi plot (Part 5, tidak diubah) ----------
def plot_feature_maps(feature_maps):
    col = 8
    row = int(feature_maps.shape[3] / col)
    ix = 1
    plt.figure(figsize=(20, 20))
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.show()

# ---------- Load gambar (sekali saja) ----------
img = load_img('Elephant.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

# ---------- Load VGG19 base model ----------
base_model = VGG19(weights='imagenet')

# ---------- Loop untuk n = 1, 2, 3, 4 ----------
for n in [1, 2, 3, 4]:
    print("\n=== Layer n = {} : {} ===".format(n, base_model.layers[n].name))

    # redefine model to output right after the hidden layer
    model = Model(inputs=base_model.inputs,
                  outputs=base_model.layers[n].output)

    # predict
    feature_maps = model.predict(img, verbose=0)
    print("Feature maps shape: ", feature_maps.shape)

    # plot
    plot_feature_maps(feature_maps)