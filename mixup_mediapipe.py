import cv2
import mediapipe as mp
import tensorflow
import keras
import numpy as np
import cv2
from turtle import color

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

import math
from sys import path
import cv2 as c
import pyautogui as p
import numpy as np
rs = p.size()
fn = input("please enter path")
fps = 60.0
fourcc= c.VideoWriter_fourcc(*'XVID')
filename="recording_mixture1.avi"
output = c.VideoWriter(filename,fourcc,fps,rs)
c.namedWindow("Live_Recording",c.WINDOW_NORMAL)
c.resizeWindow("Live_Recording",(600,400))





class Classifier:

    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        np.set_printoptions(suppress=True)
        self.model = tensorflow.keras.models.load_model(self.model_path)

        
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float64)
        self.labels_path = labelsPath
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = ["ANAND","ISHA","NONE"]
            for line in label_file:
                stripped_line = line.strip()
                self.list_labels.append(stripped_line)
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw= True, pos=(50, 50), scale=2, color = (0,255,0)):
        
        imgS = cv2.resize(img, (224, 224))
        
        image_array = np.asarray(imgS)
       
        normalized_image_array = (image_array.astype(np.float64) / 127.0) - 1

       
        self.data[0] = normalized_image_array

        
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[indexVal]),
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal

cap = cv2.VideoCapture(0)
maskClassifier = Classifier('D:/recognitionimage/my_model/keras_model.h5', 'D:/recognitionimage/my_model/labels.txt')
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # face_cas = cv2.CascadeClassifier("C/Users/HP/AppData/Local/Programs/Python/python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    
        while True:
            # ret, img = cap.read()
            ret, frame = cap.read()
            predection = maskClassifier.getPrediction(frame)
            print(predection)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            print(results.face_landmarks)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(280,110,10 ), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
            # ret,img = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
            # humans = face_cas.detectMultiScale(img)

            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)       
            cv2.imshow("Pose_estimator", image)
            if cv2.waitKey(10) & 0xff == ord("q"):
                    break

            img = results(image)
            lmList, bboxInfo = results.POSE_CONNECTIONS(img, bboxWithHands=True)
            imga =p.screenshot()
            f=np.array(imga)
            f=c.cvtColor(f ,c.COLOR_BGR2RGB)
            output.write(f)
            # f.save("D:/recognitionimage/D:\recognitionimage")
            # for (x,y,w,h) in humans:
            #   cv2.rectangle(img,(x,y), (x+w,y+h), (255,0,255),2)
        
            if bboxInfo:
                center = bboxInfo["center"]
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            

            # cv2.imshow("Image", img)
            cv2.imshow("Live_Recording",f)
            
            # if cv2.waitKey(1) & 0xff==ord("q"):
                # break

cap.release()
cv2.destroyAllWindows()
# if __name__ == "__main__":
    # main()
# else:
#     print("hello")

