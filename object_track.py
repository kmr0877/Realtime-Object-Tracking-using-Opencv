import cv2
import numpy as np
import pylab
import os
import shutil
import imageio
from collections import deque
from collections import deque
import numpy as np
import argparse

import imutils
import cv2
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
 
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=20)
counter = 0
(dX, dY) = (0, 0)
direction = ""
filename = 'test_sample1.avi'
vid = imageio.get_reader(filename,  'ffmpeg')
count = 0
for num in range(vid.get_length()):
    try:
        img = vid.get_data(num)
        
        img = cv2.resize(img,(1000,600))
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _,im =  cv2.threshold(gray_image,220,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        image, cnts, hier = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        center = None
	for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)

            # convert all coordinates floating point values to int
            box = np.int0(box)

            # draw a red 'nghien' rectangle
            cv2.drawContours(img, [box], 0, (0, 0, 255))
        # only proceed if at least one contour was found
	if len(cnts) > 1:
            
            # Change objects for finding trajectory
            # By changing the value of -2 to any other integer you can change the tracking
            # if you change it to -3 it will pickup the object which has third largest area  
            # and so on. but the area of objects will change as you move and rotate the image
            # So it will not give much accurate results
            # -2 works best because it is the second largest area. first largest area is usually 
            # the full video size.
            c = sorted(cnts, key=cv2.contourArea)[-2]
            
            # for tracking most dominant object uncomment this and comment above  line
       #     c = [contour for contour in sorted(cnts, key=cv2.contourArea)[::-1] if cv2.contourArea(contour)< 450000][0]
             
            
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(img, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)
	
def process_image(im):
    image, contours, hierarchy= cv2.findContours(im, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        
        # get the bounding rect

        x, y, w, h = cv2.boundingRect(c)
        
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(img,'(x,y)',(x-25,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    
        
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        
        # convert all coordinates floating point values to int
        box = np.int0(box)
        
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))

    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    return img

try:
    os.mkdir("images")
except:
    shutil.rmtree("images")
    os.mkdir("images")

try:
    shutil.rmtree('output.avi')
except:
    pass
filename = 'vtest.avi'
vid = imageio.get_reader(filename, 'ffmpeg')

for num in range(vid.get_length()):

	img = vid.get_data(num)
	gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	_,im =  cv2.threshold(gray_image,220,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	im = process_image(im)
	cv2.imshow('image',im)
	#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
	fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
	#w = int(vid.get(cv2.VID_PROP_FRAME_WIDTH))
	#h = int(vid.get(cv2.VID_PROP_FRAME_HEIGHT))
	#out = cv2.VideoWriter('output.avi',fourcc,20.0,(w,h))
	out = cv2.VideoWriter('output.avi',-1,20.0,(640,480))
	#vid.release()
	pylab.imsave("images/"+"image-"+str(num)+".jpg",im)

os.system('ffmpeg -framerate 25 -i images/image-%00d.jpg -r 76 -s 800x600 output.avi')




