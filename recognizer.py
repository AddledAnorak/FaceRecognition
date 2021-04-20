import cv2
import pickle
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognition_model.yml")

cap = cv2.VideoCapture(0)
labels = {}

with open('labels.pickle', "rb") as f:
    labels = pickle.load(f)

# invert the labels
labels = {value: key for (key, value) in labels.items()}

print(labels)

while True:
    ret, frame = cap.read()

    # convert it to gray_scale and get the faces
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 3, minSize = (80, 80))
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+h]
        label_id, confidence_level = recognizer.predict(roi_gray)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        if confidence_level >= 40 and confidence_level < 65:
            tabs = '\t' * label_id
            print(f"{tabs}{time.time()}:{labels[label_id]}: {confidence_level}")

    if cv2.waitKey(1) == ord('q'):
        break
    
    cv2.imshow("LIVE VIDEO FEED", frame)

cap.release()
cv2.destroyAllWindows()