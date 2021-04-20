import time
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognition_model.yml")

cap = cv2.VideoCapture(0)
labels = {}
continue_loop = True
NUM_TRIES = 5

with open('labels.pickle', "rb") as f:
    labels = pickle.load(f)

person_freq = {key: 0 for key in labels}

# invert the labels
labels = {value: key for (key, value) in labels.items()}

_ = input('ready? ')

for i in range(NUM_TRIES):
    start_time = time.time()

    while time.time() - start_time < 1.05:
        ret, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 3, minSize = (80, 80))

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+h]
            label_id, confidence_level = recognizer.predict(roi_gray)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

            if confidence_level >= 40 and confidence_level < 65:
                person_freq[labels[label_id]] += 1
    
    for person, frequency in person_freq.items():
        if frequency > 17:
            print(f"Unlocked for {person} with {frequency} detections.\t{person_freq}")
            continue_loop = False
    
    if not continue_loop:
        break

    if i == NUM_TRIES-1:
        print(f"Match Not Found.\t{person_freq}")
    else:
        print(f"Match Not Found. Trying Again\t{person_freq}")
            

cap.release()
cv2.destroyAllWindows()