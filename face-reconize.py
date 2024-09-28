import cv2
import numpy as np
import os

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recog = cv2.face.LBPHFaceRecognizer.create()
faces = []
ids = []

for i in range(1,3):
    paths = f'face{i}'

    for item in os.listdir(path=paths):

        print(f"read file: face{i}/{item}")
        img = cv2.imread(f'face{i}/{item}')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_np = np.array(gray,'uint8')
        face = detector.detectMultiScale(gray)

        for(x,y,w,h) in face:
            faces.append(img_np[y: y+h, x: x+w])
            ids.append(i)

print('start training...')
recog.train(faces, np.array(ids))
recog.save('face2.yml')
print('DONE')