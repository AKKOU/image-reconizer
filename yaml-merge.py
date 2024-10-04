import cv2
import os

# 創建 LBPHFaceRecognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 讀取第一個 YAML 檔案
firstFile = input("Firstfile name:")
recognizer.read(f'face_yml/{firstFile}')

# 讀取並合併其他 YAML 檔案
print("Start merging...")
for item in os.listdir(path="face_yml/"):
    if item.endswith('.yaml') and item != firstFile:
        temp_recognizer = cv2.face.LBPHFaceRecognizer_create()
        temp_recognizer.read(os.path.join("face_yml/", item))
        
        # 獲取臨時模型的訓練資料
        labels, histograms = temp_recognizer.getHistograms()
        
        # 更新主模型
        recognizer.update(histograms, labels)

# 保存合併後的模型
recognizer.write('face_yml/merged.yml')
print("YML merge done!")
a = input("Press any key to exit...")
