import pandas as pd
import streamlit as st
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
import cv2
import tempfile
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input
from  tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def read_img(img): 
    image = cv2.imread(img)
    image = cv2.resize(image, (224,224))

    return image

st.title('Prédiction Chien')

model = load_model("./model.h5")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    st.image(opencv_image, channels="BGR")

    dog_classes = os.listdir('./Images/')

    X = []
    y = []

    X.append(opencv_image)
    for breed in dog_classes:
        y.append(breed)

    encoder = LabelBinarizer()

    X = np.array(X)
    y = encoder.fit_transform(np.array(y))

    if(st.button('Prédiction')):

        predictions = model.predict(X)

        label_predictions = encoder.inverse_transform(predictions)

        st.write(label_predictions)