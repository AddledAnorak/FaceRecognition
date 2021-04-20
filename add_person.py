import os
import cv2
import numpy as np
from os import sys

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
person = sys.argv[1]
base_dir = os.path.dirname(os.path.abspath(__file__))

for i, filename in enumerate(os.listdir('raw_faces')):
    gray_img = cv2.imread(f"{base_dir}\\raw_faces\\{filename}", 0)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 3, minSize = (80, 80))

    for (x, y, w, h) in faces:
        print(f'writing img {filename}')
        cv2.imwrite(f"{base_dir}\\known_faces\\{person}_{i}.jpg", gray_img[y:y+h, x:x+h])