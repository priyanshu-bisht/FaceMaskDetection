import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class fmd:
    def __init__(self):
        self.model = tf.keras.models.load_model('models/mask01.h5')
        self.cas = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    def predict(self, image):
        return int(np.round(self.model.predict(np.expand_dims(image,0))[0][0]))
    
    def detectV2(self,image):
        faces = self.cas.detectMultiScale(image, 1.1,4)
        font = cv.FONT_HERSHEY_PLAIN 
        color = (255, 255, 255)
        classes = ['WithMask','WithoutMask']
        i = 0 
        for (x,y,w,h) in faces:
            new_img = cv.resize(image[y:y+h, x:x+w], (50,50))
            ans = self.predict(new_img/255)
            cv.rectangle(image, (x, y+h),(x+w,y+h+30) , (0,255*(1-ans),255*ans), thickness=-1)
            cv.rectangle(image, (x,y),(x+w,y+h), (0,255*(1-ans),255*ans),2)
            cv.putText(image, classes[ans]  , (x+5,y+h+25), font, 2, color,2)
        return image

    def start(self):
        print('press q to quit')
        cap = cv.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            img = self.detectV2(frame)
            cv.imshow('Video', img)    
            if cv.waitKey(1) & 0xFF == ord('q'):
                break        
        cap.release()
        cv.destroyAllWindows()

f = fmd()
f.start()

