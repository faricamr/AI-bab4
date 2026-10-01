# Exercise 4.17
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

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
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    plt.show()

model = VGG19()
n = 2
model_feat = Model(inputs=model.inputs, outputs=model.layers[n].output)

try:
    img = load_img('Elephant.jpg', target_size=(224, 224))
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature_maps = model_feat.predict(img)
    plot_feature_maps(feature_maps)
except Exception as e:
    print("Membutuhkan file Elephant.jpg.", e)
