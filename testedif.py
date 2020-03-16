import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from umucv.umucv.kalman import kalman, ukf
import umucv.umucv.htrans as ht

REDU = 8
def rgbh(xs,mask):
    def normhist(x): return x / np.sum(x)

    def h(rgb):
        return cv2.calcHist([rgb]
                                , [0,1,2]
                                , imCropMask
                                ,[256//REDU, 256//REDU, 256//REDU]
                                , [0,256] + [0,256] + [0,2561]
                            )
    return normhist(sum(map(h,xs)))

def smooth(s,x):
    return gaussian_filter(x,s,mode='constant')

cap = cv2.VideoCapture("video.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2(500,60, True)
kernel = np.ones((3,3) , np.uint8)
crop = False
camshift = False

termination = (cv2.TERM_CRITERIA_EPS    | cv2.TERM_CRITERIA_COUNT,10,1)

font = cv2.FONT_HERSHEY_SIMPLEX
pause = False
degree = np.pi/180
a = np.array([0,900])

fps = 120
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


while True:
    cv2.waitKey(40)
    key = cv2.waitKey(1) & 0xFF
    if key== ord("c") : crop = True
    if key== ord("p") : P = np.diag([100,100,100,100])**2
    if key== 27: break
    if key== ord(" ") : pause = not pause
    if (pause) : continue



    ret, frame = cap.read()
    #
    #frame = cv2.resize(frame, (1366,768))
    
    fgmask = fgbg.apply(frame)

    fgmask = cv2.erode(fgmask,kernel,iterations = 1)
    fgmask = cv2.medianBlur(fgmask, 3)
    fgmask = cv2.dilate(fgmask,kernel,iterations = 2)
    
    fgmask = (fgmask > 200).astype(np.uint8)*255
    colorMask = cv2.bitwise_and(frame, frame, mask = fgmask)
    
    if(crop):
        fromCenter= False
        img = colorMask
        r = cv2.selectROI(img,fromCenter)
        imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        crop = False 
        camshift = True
        imCropMask = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
        ret,imCropMask = cv2.threshold(imCropMask,30,255,cv2.THRESH_BINARY)    
        his = smooth(1,rgbh([imCrop],imCropMask))
        
        roiBox = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        cv2.destroyWindow("ROI selector")
    
    if(camshift):

        cv2.putText(frame,"Center roiBox",(0,10),font,0.5,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,"Estimated Pos", (0,30),font,0.5,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,"Prediction",(0,50),font, 0.5, (0,0,255),2,cv2.LINE_AA)

        rgbr = np.floor_divide(colorMask, REDU)
        r,g,b = rgbr.transpose(2,0,1)
        l = his[r,g,b]
        maxl = l.max()
        aa= np.clip((l*l/maxl*255),0,255).astype(np.uint8)

        (rb,roiBox) = cv2.CamShift(l, roiBox, termination)
        cv2.ellipse(frame, rb, (0, 255, 0), 2)

        xo = int (roiBox[0]+roiBox[2]/2)
        yo = int (roiBox[1]+roiBox[3]/2)
        error=(roiBox[3])

        if(yo<error or fgmask.sum()<50):
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

    
    
    
    cv2.imshow("mask",colorMask)
    cv2.imshow("fg", fgmask)
    cv2.imshow("Frame", frame)



    #https://github.com/pabsaura/Prediction-of-Trajectory-with-kalman-filter-and-open-cv
