import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 1

while True:
    key = input('>>> ')

    if key == "q":
        break

    # capture the BGR frame
    ret, frame = cap.read()

    # convert it to gray_scale and get the faces
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 3, minSize = (80, 80))

    # print rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.imwrite(f"known_faces/shoan_{count}.jpg", gray_frame[y:y+h, x:x+w])
        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

# release the camera and destroy the windows
cap.release()
cv2.destroyAllWindows()