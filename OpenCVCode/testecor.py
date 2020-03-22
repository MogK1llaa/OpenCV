from collections import deque
from imutils.video import VideoStream
from scipy.ndimage.filters import gaussian_filter
from umucv.umucv.kalman import kalman, ukf
import umucv.umucv.htrans as ht
import numpy as np
import argparse
import cv2
import imutils
import time

degree = np.pi/180
a = np.array([0,900])

fps = 60
dt = 1/fps
t = np.arange(0,2.01,dt)
noise = 3

F = np.array(
    [1, 0, dt, 0,
     0, 1, 0, dt,
     0, 0, 1, 0,
     0, 0, 0, 1 ]).reshape(4,4)

B = np.array(
    [dt**2/2, 0,
     0,       dt**2/2,
     dt,      0,
     0,       dt ]).reshape(4,2)

H = np.array(
    [1, 0, 0, 0,
     0, 1, 0, 0]).reshape(2,4)

mu = np.array([0,0,0,0])
P = np.diag([1000,1000,1000,1000])**2
#res = [(mu,P,mu)]
res=[]
N = 15

sigmaM = 0.0001
sigmaZ = 3* noise

Q = sigmaM**2 * np.eye(4)
R = sigmaZ**2 * np.eye(2)

listCenterX=[]
listCenterY=[]
listpoints=[]

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("sample2.mp4")
#cap = cv2.VideoCapture("video.mp4")
low_red = np.array([161, 100, 84])
high_red = np.array([179, 255, 255])

pts = deque(maxlen=32)
counter = 0
(dX,dY)= (0,0)

while True:
    _, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
   
    blur = cv2.GaussianBlur(frame, (11,11), 0)
    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red_mask = cv2.erode(red_mask, None, iterations=2)
    red_mask = cv2.dilate(red_mask, None, iterations=2)

    cnts = cv2.findContours(red_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)
    
        # kalman filter

        xo = int(M["m10"] / M["m00"])
        yo = int(M["m01"] / M["m00"])

        #yo<error or
        if(frame.sum()<50):
            mu,P,pred= kalman(mu,P,F,Q,B,a,None,H,R)
            m="None"
            mm=False
        else:
            mu,P,pred=kalman(mu,P,F,Q,B,a,np.array([xo,yo]),H,R)
            m="normal"
            mm=True
        
        if(mm) :
            listCenterX.append(xo)
            listCenterY.append(yo)
        
        listpoints.append((xo,yo,m))
        res += [(mu,P)]

        mu2 = mu
        P2 = P
        res2 = []

        for _ in range(fps*2):
            mu2,P2,pred2=kalman(mu2,P2,F,Q,B,a,None,H,R)
            res2 += [(mu2,P2)]
        
        xe = [mu[0] for mu,_ in res]
        xu = [2*np.sqrt(P[0,0]) for _,P in res]
        ye = [mu[1] for mu,_ in res]
        yu = [2*np.sqrt(P[1,1]) for _,P in res]

        xp = [mu2[0] for mu2,_ in res2]
        yp = [mu2[1] for mu2,_ in res2]
        xpu = [2*np.sqrt(P[0,0]) for _,P in res2]
        ypu = [2*np.sqrt(P[1,1]) for _,P in res2]

        for n in range(len(listCenterX)):
            cv2.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0,255,0),-1)
        
        for n in [-1]:
            incertidumbre = (xu[n]+yu[n])/2
            cv2.circle(frame, (int(xe[n]),int(ye[n])),int(incertidumbre),(255,255,0),1)

        for n in range(len(xp)):
            incertidumbreP=(xpu[n]+ypu[n])/2
            cv2.circle(frame,(int(xp[n]), int(yp[n])),int(incertidumbreP),(0,0,255))

        print("Lista de pontos\n")
        for n in range(len(listpoints)):
            print(listpoints[n])

        if(len(listCenterY)>4):
            if( (listCenterY[-5] < listCenterY[-4]) and (listCenterY[-4] < listCenterY[-3]) 
            and (listCenterY[-3] > listCenterY[-2]) and (listCenterY[-2] > listCenterY[-1])):
                print ("REBOTE")
                listCenterY=[]
                listCenterX=[]
                listpoints=[]
                res=[]
                mu = np.array([0,0,0,0])
                P = np.diag([100,100,100,100])**2

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(60)
    if key == 27:
        break
    if key == 32:
        time.sleep(20)
cv2.destroyAllWindows()