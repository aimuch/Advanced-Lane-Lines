
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import sys

# callback function
def nothing(x):
    pass

# import image 
inputImageName = 'test_videos/solidYellowLeft/143.jpg'

try:
    img = cv2.imread(inputImageName)  # BRG
    WindowName = 'Color Select V1.0'
    cv2.namedWindow(WindowName)
    cv2.imshow(WindowName,img)
except Exception as err:   
    print ('Exception: ', err)
    print("Open file %s failed!"%(inputImageName))
    sys.exit(-1)

# create trackbars for color change
WindowName_1_L = 'CH 1 Down'
WindowName_1_R = 'CH 1 UP'
WindowName_2_L = 'CH 2 Down'
WindowName_2_R = 'CH 2 UP'
WindowName_3_L = 'CH 3 Down'
WindowName_3_R = 'CH 3 UP'

cv2.createTrackbar(WindowName_1_L,WindowName,0,255,nothing)
cv2.createTrackbar(WindowName_1_R,WindowName,0,255,nothing)
cv2.setTrackbarPos(WindowName_1_L,WindowName,0)
cv2.setTrackbarPos(WindowName_1_R,WindowName,255)

cv2.createTrackbar(WindowName_2_L,WindowName,0,255,nothing)
cv2.createTrackbar(WindowName_2_R,WindowName,0,255,nothing)
cv2.setTrackbarPos(WindowName_2_L,WindowName,0)
cv2.setTrackbarPos(WindowName_2_R,WindowName,255)

cv2.createTrackbar(WindowName_3_L,WindowName,0,255,nothing)
cv2.createTrackbar(WindowName_3_R,WindowName,0,255,nothing)
cv2.setTrackbarPos(WindowName_3_L,WindowName,0)
cv2.setTrackbarPos(WindowName_3_R,WindowName,255)

switch = '0 : RGB \n1 : HSV \n2 : HSL'
cv2.createTrackbar(switch,WindowName,0,2,nothing)

def colorselect(image,ch1,ch1up,ch2,ch2up,ch3,ch3up):
    if ch1<ch1up and ch2<ch2up and ch3<ch3up:
        mode = cv2.getTrackbarPos(switch,WindowName)
        modeDict = {0:cv2.COLOR_BGR2RGB, 1:cv2.COLOR_BGR2HSV, 2:cv2.COLOR_BGR2HLS}
        colormode = modeDict[mode]

        convertedImage = cv2.cvtColor(image, colormode)
        lower_color = np.array([ch1, ch2, ch3]) 
        upper_color = np.array([ch1up, ch2up, ch3up]) 
        color_mask = cv2.inRange(convertedImage, lower_color, upper_color)

        dst = cv2.bitwise_and(image, image, mask = color_mask)
        return dst
    else:
        return image

if __name__ == '__main__':
    while(True):
        # Push ESC exit
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        ch1 = cv2.getTrackbarPos(WindowName_1_L,WindowName)
        ch1up= cv2.getTrackbarPos(WindowName_1_R,WindowName)
        ch2 = cv2.getTrackbarPos(WindowName_2_L,WindowName)
        ch2up= cv2.getTrackbarPos(WindowName_2_R,WindowName)
        ch3 = cv2.getTrackbarPos(WindowName_3_L,WindowName)
        ch3up= cv2.getTrackbarPos(WindowName_3_R,WindowName)

        filtedimg = colorselect(img,ch1,ch1up,ch2,ch2up,ch3,ch3up)

        # Display the image 
        cv2.imshow(WindowName,filtedimg)                

    cv2.destroyAllWindows()

