import mediapipe as mp
import cv2
import handTrackingmodule as htm
import pynput
from pynput.mouse import Controller,Button
import time
import screeninfo
import numpy as np

mouse = Controller()
cap = cv2.VideoCapture(0)
wcam = 720
hcam = 600
prevTime=0
red_size = 50

#smoothness values
smoothness = 8
prev_x , prev_y = 0,0
c_x , c_y = 0,0


#setting window size
cap.set(3,wcam)
cap.set(4,hcam)

#getting screen size
screen = screeninfo.get_monitors()[0]
hscr, wscr = screen.height,screen.width

detector = htm.handDetector(False,1,0,0)

while True:
    #reading frames
    suc , img = cap.read()

    #detecting hands and getting landmarks list
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img)

    #drawing rectangle for reduced window
    cv2.rectangle(img,(red_size,red_size),(wcam-red_size,hcam-red_size),(255,0,255),3)

    if lmList:
        #getting coordinates of index and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        
        #finding which fingers are up
        up = detector.fingersUp()
        
        #checking that only index finger is up
        if up[1]==1 and up[2]==0:
            #converting ind finger coordinates to screen size
            x3 = np.interp(x1,(red_size,wcam-red_size),(0,wscr))
            y3 = np.interp(y1,(red_size,hcam-red_size),(0,hscr))

            #sending these coordinates to mouse
            c_x = prev_x + (x3-prev_x)/smoothness
            c_y = prev_y + (y3-prev_y)/smoothness

            cur_x , cur_y = mouse.position
            mouse.move(wscr-c_x-cur_x,c_y-cur_y)
            prev_x, prev_y = c_x,c_y
            cv2.circle(img,(x1,y1),15,(255,255,0),cv2.FILLED)
      
        #checking that both index and middle fingers are up --for clicking
        if up[1]==1 and up[2]==1:
            length , img , line = detector.findDistance(8,12,img)
            if length<40:
                cv2.circle(img,(line[4],line[5]),15,(0,255,0),cv2.FILLED)
                #left click
                mouse.click(Button.left)



    #getting the frame rate and showing it
    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime=curTime
    cv2.putText(img,str(int(fps)),(30,60),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,50),3)

 
    #displaying the cam feed
    cv2.imshow("image",img)

    #closing window when l is pressed
    if cv2.waitKey(1)==ord('l'):
        break
    