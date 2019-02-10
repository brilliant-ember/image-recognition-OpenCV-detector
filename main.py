import numpy as np
import cv2
import requests 
import threading
from time import sleep

# img = cv2.imread('watch.jpeg',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

####Video capture
# cap = cv2.VideoCapture(0)
# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.line(gray,(0,0),(150,150),(255,255,255),15)

#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

###Motion detection
# cap = cv2.VideoCapture(0)
# fgbg = cv2.createBackgroundSubtractorMOG2() #non action reduction
# while(1):
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
 
#     cv2.imshow('fgmask',frame)
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()

detected = False
runDetector = True
#pres ESC to exit Mohammed!! dont forget CTRL + C wont do close the program
def openCV():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    upper_body = cv2.CascadeClassifier("haarcascade_upperbody.xml")

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CV_CAP_PROP_FPS, 10)

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # IMREAD_GRAYSCALE
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        body = body_cascade.detectMultiScale(gray)
        upperBody = upper_body.detectMultiScale(gray)

        ##### body
        # for (x,y,w,h), (x1,y1,w1,h1) in zip(upperBody, body):
        #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        # ##### face
        for (x,y,w,h) in faces:
            global detected
            global runDetector
            
            detected = True
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

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
            print("Detected")
            detected = False
        else:
            sleep(2)
            if not detected:
                responce = requests.post("https://xbackend.appspot.com/fallCamera", json={"status":"true"})
                if responce.status_code == 200:
                    print("sent to the server")
                    print("not detected for a few seconds")

t1 = threading.Thread(target=openCV)
t1.daemon = True
t2 = threading.Thread(target=mointerDetectionFlag)
t2.daemon = True
t1.start()
t2.start()
t1.join()
t2.join()