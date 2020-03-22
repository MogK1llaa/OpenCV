import cv2
import numpy as np


# Capturing video from webcam:
cap = cv2.VideoCapture(0)

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Handles the mirroring of the current frame
    frame1 = cv2.flip(frame1,1)
    frame2 = cv2.flip(frame2,1)


    diff = cv2.absdiff(frame1,frame2)
 
    
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

   
    blur = cv2.GaussianBlur(gray, (5,5), 0)

   
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
   
    
    dilated = cv2.dilate(thresh, None, iterations=3)
   
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    #currentFrame += 1

# When everything done, release the capture
cap.release()
cap1.release()
cv2.destroyAllWindows()