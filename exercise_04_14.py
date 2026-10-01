# Exercise 4.14 
import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import (
    vgg16, resnet50, mobilenet_v2, densenet, inception_v3
)

MODEL_NAME = 'mobilenetv2'

if MODEL_NAME == 'vgg16':
    model = vgg16.VGG16(weights='imagenet')
    preprocess_input = vgg16.preprocess_input
    sz = 224
    display_name = 'VGG16'

elif MODEL_NAME == 'resnet50':
    model = resnet50.ResNet50(weights='imagenet')
    preprocess_input = resnet50.preprocess_input
    sz = 224
    display_name = 'ResNet50'

elif MODEL_NAME == 'mobilenetv2':
    model = mobilenet_v2.MobileNetV2(weights='imagenet')
    preprocess_input = mobilenet_v2.preprocess_input
    sz = 224
    display_name = 'MobileNetV2'

elif MODEL_NAME == 'densenet201':
    model = densenet.DenseNet201(weights='imagenet')
    preprocess_input = densenet.preprocess_input
    sz = 224
    display_name = 'DenseNet201'

elif MODEL_NAME == 'inceptionv3':
    model = inception_v3.InceptionV3(weights='imagenet')
    preprocess_input = inception_v3.preprocess_input
    sz = 299
    display_name = 'InceptionV3'

else:
    raise ValueError("Model tidak dikenal: " + MODEL_NAME)

print("Model dipilih:", display_name)
model.summary()

camera = cv2.VideoCapture(0)
image_size = sz

while True:
    ret, cam_frame = camera.read()
    frame = cv2.resize(cam_frame, (image_size, image_size))
    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    preds = model.predict(image, verbose=0)
    label = decode_predictions(preds)

    cv2.putText(cam_frame, "{}: {}, {:.1f}".format(
        display_name, label[0][0][1], label[0][0][2]),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", cam_frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()