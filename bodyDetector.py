import numpy as np
import cv2
import requests 
import threading
from time import sleep

detected = False
runDetector = True

#pres ESC to exit Mohammed!! dont forget CTRL + C wont do close the program
def openCV():
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    upper_body = cv2.CascadeClassifier("haarcascade_upperbody.xml")

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CV_CAP_PROP_FPS, 10)
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # IMREAD_GRAYSCALE
        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        body = body_cascade.detectMultiScale(gray)
        upperBody = upper_body.detectMultiScale(gray)

        ##### body
        for (x,y,w,h), (x1,y1,w1,h1) in zip(upperBody, body):
            global detected
            global runDetector
            detected = True
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            runDetector = False
            break

    cap.release()
    cv2.destroyAllWindows()

def mointerDetectionFlag():
    global detected
    global runDetector
    
    while runDetector:
        sleep(0.5)
        if detected:
            print("Detected a body")
            detected = False
            responce = requests.post("https://xbackend.appspot.com/fallCamera", json={"status":"true"})
            if responce.status_code == 200:
                print("sent to the server")
        else:
            sleep(2)
            if not detected:
                print("not detected for a few seconds did you fall?")


t1 = threading.Thread(target=openCV)
t1.daemon = True
t2 = threading.Thread(target=mointerDetectionFlag)
t2.daemon = True
t1.start()
t2.start()
t1.join()
t2.join()