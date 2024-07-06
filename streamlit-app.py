import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import io

import numpy as np
import cv2
import os
import pickle
import imutils
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from imutils.video import VideoStream

net = cv2.dnn.readNetFromCaffe("./deploy.prototxt.txt", "./res10_300x300_ssd_iter_140000.caffemodel")
model = load_model("liveness.keras")
le = pickle.loads(open("le.pickle", "rb").read())


"""
# Computer Vision
## Liveness Detection

O detector de Liveness (Vivacidade) tem por objetivo estabelecer um índice que atesta o quão 
confiável é a imagem obtida pela câmera.
Imagens estáticas, provindas de fotos manipuladas, são os principais focos de fraude neste tipo de validação.
Um modelo de classificação deve ser capaz de ler uma imagem da webcam, classificá-la como (live ou não) e 
exibir sua probabilidade da classe de predição.

"""

uploaded_file = st.file_uploader('Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    st.image(imagem, channels="BGR")


camera = st.camera_input("Tire sua foto", help="Lembre-se de permitir ao seu navegador o acesso a sua câmera.")

if camera is not None:
    bytes_data = camera.getvalue()
    imagem = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


if camera or uploaded_file:
    confidence = 0
    with st.spinner('Classificando imagem...'):
        image = imutils.resize(imagem, width=600)
        (h, w) = imagem.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.50:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                face = imagem[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le.classes_[j]
                if(str(label).endswith("Real")):
                    st.success("Probabilidade da foto ser real é de {:.2f}%".format(confidence*100))
                    break
                else:
                    st.error("Probabilidade da foto não ser real é de {:.2f}%".format(confidence*100))
                    break
            else:
                st.error("Nenhum rosto encontrado na imagem")
                break