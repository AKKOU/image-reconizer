import cv2
import numpy as np
import os
import json

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recog = cv2.face.LBPHFaceRecognizer.create()
faces = []
ids = []

# input data
face_id = input("Enter face ID:")
with open('./face-list.json', 'r') as file:
    json_data = json.load(file)

paths = f'faces_data/{json_data[face_id]}'

for item in os.listdir(path=paths):

    print(f"read file: {paths}/{item}")
    img = cv2.imread(f'{paths}/{item}')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')
    face = detector.detectMultiScale(gray)

    for(x,y,w,h) in face:
        faces.append(img_np[y: y+h, x: x+w])
        ids.append(int(face_id))

print('start training...')
recog.train(faces, np.array(ids))
recog.save(f'face_yml/{json_data[face_id]}.yml')
print('DONE')