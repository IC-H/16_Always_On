import urllib2
from PIL import Image
import cv2
import numpy as np
import math
import pytesseract
from gtts import gTTS
import sys
import os
import combine_4th as cm



cm.handle_video()

r1=0

img = cv2.imread('a2.png',0) #Rotating
rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),r1,1) #rotate(r1)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imwrite('sss.png',dst)
speach = "hello hi " + pytesseract.image_to_string(Image.open('sss.png')).strip() 
print speach
tts = gTTS(text=speach, lang='en') #gTTS engine speach to english
tts.save('b.mp3')
os.system('mpg321 b.mp3') #playing the mp3file

#os.remove('sss.png') #delete the pngfile
#os.remove('alwayson.mp3') #delete the mp3file


