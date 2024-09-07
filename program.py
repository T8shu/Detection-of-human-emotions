'''''
import numpy as np
import cv2 
from deepface import DeepFace
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

while video.isOpened:
    _,frame = video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)

    for x,y,w,h in face:
        img = cv2.rectangle(frame,(x+w,y+h),(0,0,255),1)
        try:
            analyze = DeepFace.analyze(frame,actions=['emotion'])
            print(analyze[0]['dominant_emotion'])
        except:
            print("no face")


    cv2.imshow('video',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()               

'''
# imgpath = 'img6.jpg'
# image = cv2.imread(imgpath)

# analyze = DeepFace.analyze(image, actions = ['emotion'])
# # print(analyze)
# print(analyze[0]['dominant_emotion'])

import numpy as np
import cv2 
from deepface import DeepFace

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize video capture from webcam
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Analyze the face for emotions
        try:
            # Crop the face from the frame for analysis
            face_img = frame[y:y+h, x:x+w]
            analyze = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            print(analyze[0]['dominant_emotion'])
        except Exception as e:
            print("No face detected or error:", e)

    # Display the video frame
    cv2.imshow('Video', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video.release()
cv2.destroyAllWindows()