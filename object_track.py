import cv2
import numpy as np
import pylab
import os
import shutil
import imageio

def process_image(im):
    image, contours, hier = cv2.findContours(im, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)

    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    for c in contours:
        
        # get the bounding rect
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

    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    return img

try:
    os.mkdir("images")
except:
    shutil.rmtree("images")
    os.mkdir("images")


# In[13]:

try:
    shutil.rmtree('foo.avi')
except:
    pass


# In[14]:

filename = 'people-walking.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')

for num in range(vid.get_length()):
    img = vid.get_data(num)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _,im =  cv2.threshold(gray_image,220,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    im = process_image(im)
    pylab.imsave("images/"+"image-"+str(num)+".jpg",im)


# In[15]:

os.system('ffmpeg -framerate 25 -i images/image-%00d.jpg -r 76 -s 800x600 foo.avi')






