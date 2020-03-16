import cv2
import numpy as np


# Capturing video from webcam:
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    ret, frame11 = cap1.read()
    ret, frame22 = cap1.read()

    # Handles the mirroring of the current frame
    frame1 = cv2.flip(frame1,1)
    frame2 = cv2.flip(frame2,1)
    frame11 = cv2.flip(frame11,1)
    frame22 = cv2.flip(frame22,1)

    diff = cv2.absdiff(frame1,frame2)
    diff1 = cv2.absdiff(frame11,frame22)
    
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
   
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    blur1 = cv2.GaussianBlur(gray1, (5,5), 0)
   
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    _, thresh1 = cv2.threshold(blur1, 20, 255, cv2.THRESH_BINARY)
    
    dilated = cv2.dilate(thresh, None, iterations=3)
    dilated1 = cv2.dilate(thresh1, None, iterations=3)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours1, _ = cv2.findContours(dilated1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x,y,w,h)= cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 6000:
            continue
        cv2.rectangle(frame1, (x,y), (x+w,y+h),(0,255,0),2)
        cv2.putText(frame1, "Coordenadas: x= {} y= {}".format(x,y), (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
       
    
    cv2.drawContours(frame11, contours1, -1, (0,255,0),2)


    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1280,720)
    cv2.imshow('frame',frame1)
    frame1 = frame2
    ret, frame2 = cap.read()


    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('camera', 640,480)
    cv2.imshow('camera',frame11)
    frame11 = frame22
    ret, frame22 = cap1.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    #currentFrame += 1

# When everything done, release the capture
cap.release()
cap1.release()
cv2.destroyAllWindows()