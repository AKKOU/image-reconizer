import cv2
import numpy as np
import mss
import mss.tools
import os
import json

print("Starting up the process...")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recog = cv2.face.LBPHFaceRecognizer.create()

if os.path.exists("face_yml/merged.yml"):
    recog.read(f"face_yml/merged.yml")  
    print("Loading merged.yml...")
else:
    Fname = input("Can't find merged.yml, please enter the YML file manually: ")
    recog.read(f"face_yml/{Fname}")  
    print(f"Loading {Fname}...")

with mss.mss() as sct:
    monitor = sct.monitors[3]

    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray)

        with open('./face-list.json', 'r') as file:
            name = json.load(file)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            idnum,confidence = recog.predict(gray[y:y+h,x:x+w])
            if confidence < 60:
                text = name[str(idnum)]
            else:
                text = "???"
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Screen', img)

        if cv2.waitKey(5) == ord('q'):
            break

cv2.destroyAllWindows()
