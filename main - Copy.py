import numpy as np
import cv2
import matplotlib.pyplot as plt

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

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
upper_body = cv2.CascadeClassifier("haarcascade_upperbody.xml")

cap = cv2.VideoCapture(0)
# cap.set(cv2.CV_CAP_PROP_FPS, 10)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    body = body_cascade.detectMultiScale(gray)
    upperBody = upper_body.detectMultiScale(gray)

    ##### body
    # for (x,y,w,h), (x1,y1,w1,h1) in zip(upperBody, body):
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # ##### face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()