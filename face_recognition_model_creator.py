import pickle
import cv2
import numpy as np
from PIL import Image
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

train_x = []
train_y = []
label_ids = {}
next_id = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()

for filename in os.listdir('known_faces'):
    if not filename.endswith('.jpg') and not filename.endswith('.png'):
        raise Exception(f'file {filename} is not an image file!')

    person_name = filename.split('_')[0]
    if not person_name in label_ids:
        label_ids[person_name] = next_id
        next_id += 1
    
    train_y.append(label_ids[person_name])

    numpy_image = np.array(Image.open(base_dir + '\\known_faces\\' + filename), "uint8")
    train_x.append(numpy_image)


with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# train and save the model
recognizer.train(train_x, np.array(train_y))
recognizer.save("recognition_model.yml")