import cv2
import numpy as np
import time

cv2.namedWindow('Video') #name a new window
vc = cv2.VideoCapture(0) #capture a video from default camera (camera 0,--> nb this is the rear camera on a mobile)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')    #haar cascade face
eye_cascade = cv2.CascadeClassifier('right_eye.xml')  #haar cascade eyes
the_face = None

if vc.isOpened(): # try to get the first frame
	rval, frame = vc.read()
else:
	rval = False #stop if no frames

newtime=time.time()	
oldtime = counter = counter2 = 0	
	
while rval:

	#Create FPS count
	newtime = time.time()
	counter+=1
	if ((newtime - oldtime) >= 1):
		oldtime = time.time()
		counter2 = counter
		counter = 0
		
		height, width = frame.shape[:2]
	
	#SHOW FPS COUNT:
	
	global the_face
	rval, frame = vc.read()
	key = cv2.waitKey(20)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #make gray-scale image
	
	cv2.putText(frame,"FPS: %r"%counter2,(width/14,height - height/8), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

	
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		cv2.rectangle(frame,(x+(w/6),y+(h/4)),(x+(w/2),y+(h-(h/2))),(255,0,255),2)
		roi_eyes_right_colour = frame[y+(h/4):y+(h-(h/2)), x+w/6:x + w/2]
		#roi_eyes_left = roi_gray[y+(h/3):y+(h-(h/2)), x+w/6:x + w/2]
		#roi_eyes_right = roi_gray[y+(h/6):y+(h-(h/2)), x+w/2:x + w - w/6]
		
		
		
		eye_frame_gray = cv2.cvtColor(roi_eyes_right_colour, cv2.COLOR_BGR2GRAY)
		eyes = eye_cascade.detectMultiScale(eye_frame_gray) #detect face
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_eyes_right_colour, (ex+ew/2,ey+eh/2),(ex+ew/2,ey+eh/2),(255,255,255),2) # Just detectes the top of the eye orbit????
			cv2.rectangle(roi_eyes_right_colour,(ex,ey+eh/2),(ex+ew,ey+eh),(255,255,0),2)
			
			#FIND SCHLEA
			schlea_frame = cv2.GaussianBlur(eye_frame_gray[ey+eh/2:ey+eh, ex:ex+ew],(5,5),0.01)
			schlea_frame = cv2.equalizeHist(schlea_frame)
			#_,schlea_frame = cv2.threshold(schlea_frame,35,200,cv2.THRESH_BINARY_INV)
			
			
			(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(schlea_frame)
			cv2.circle(roi_eyes_right_colour[ey+eh/2:ey+eh, ex:ex+ew], maxLoc, 2, (255, 0, 0), 1)
			cv2.circle(roi_eyes_right_colour[ey+eh/2:ey+eh, ex:ex+ew], minLoc, 2, (255, 255, 255), 1)
			#canny = cv2.Canny(eye_frame,5,150)
			
       #resize image to be 4 times the size of the original
			
			
			
			
			
			cv2.rectangle(roi_eyes_right_colour,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #detect eyes
			
			
			mask = cv2.inRange(schlea_frame, 0, 30)
			
			
			circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,param1=100,param2=10,minRadius=0,maxRadius=20) 
			if circles is None:
				print "No  Circles"
			else:
				circles = np.round(circles[0,:]).astype("int")
				for (cx,cy,cr) in circles:
					cv2.circle(schlea_frame, (cx,cy), cr, (255,255,0),1)
			

			# Bitwise-AND mask and original image
			#res = cv2.bitwise_and(frame,frame, mask= mask)

			#cv2.imshow('frame',frame)
			mask = cv2.resize(mask, (0,0), fx=8, fy=8) 
			
			
			
			
			
			cv2.imshow('mask',mask)
			#cv2.imshow('res',res)
			
			
			
			"""
			corners = cv2.goodFeaturesToTrack(schlea_frame,5,0.1,10)
			corners = np.int0(corners)
			for i in corners:
				x,y = i.ravel(order='K')
				cv2.circle(schlea_frame,(x,y),3,(0,0,0),-1)
			"""	

       #resize image to be 4 times the size of the original
			schlea_frame = cv2.resize(schlea_frame, (0,0), fx=4, fy=4) 
			cv2.imshow("eye",schlea_frame)
			roi_eyes_right_colour = cv2.resize(roi_eyes_right_colour, (0,0), fx=8, fy=8) 
			cv2.imshow("eyes?",roi_eyes_right_colour)
	cv2.imshow('Video', frame) #show what is found
	
	#Note: This will only 'continue' the video feed while a face AND eyes are detected
	
	
	if key == 27: # exit on ESC
		break
vc.release()
cv2.destroyWindow('Video')