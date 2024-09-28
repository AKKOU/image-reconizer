import cv2
import numpy as np
import mss
import mss.tools
import os

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recog = cv2.face.LBPHFaceRecognizer.create()

for item in os.listdir(path="face_yml"):
        recog.read(f"face_yml/{item}")  

with mss.mss() as sct:
    monitor = sct.monitors[3]

    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray)

        name = {
            '1': "Lai",
            '2': "AsiaTon",
            '3': "Tim Cook",
            '4': "Kojima",
            '5': "Tsai",
            '6': "Kuo",
            '7': "Xi"
        }

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
