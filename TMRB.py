import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import os
import urllib2
from PIL import Image
import pytesseract
from gtts import gTTS
import sys
from combine_4th import *


def for_Testing():
	os.system('sudo modprobe bcm2835-v4l2')
	cap = cv2.VideoCapture(0)
	#cap.set(3, 360)
	#cap.set(4, 240)
	num_c  = 0
	point_c = [(0, 0), (0, 0)]
	countFull = 0
	CaptureFlag = False
	PS = 0 #previous state
	pv = []
	''' Main Function'''
	theta = test_learn()
	ret, frame = cap.read() # 2592 * 1944

	start_line = "Now Loading...."
	tts = gTTS(text=start_line, lang='en')
	tts.save('greeting.mp3')
	os.system('mpg321 greeting.mp3')

	time.sleep(2)

	while True:
		ret, frame = cap.read() # 2592 * 1944
		frame2 = frame.copy() # copy frame 
		
		'''Color Filter for finding hands '''
		
		hand_filter_down 	= np.array([0, 133, 77]) #YCrCb Filter low Limit
		hand_filter_up	= np.array([255, 173, 127]) #YCrCb Filter Up Limit
		
		hand_filter = handFilter(frame, hand_filter_up, hand_filter_down)
		hand = hand_filter.copy()
		
		st, con, hir = cv2.findContours(hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		coor = findEndPoints(con, frame2, theta)
		
		print coor
		
		countForCoor = len(coor) - 1
		
		if (countForCoor == 1 ) & (PS == 0) & (coor[0]):
			if ( point_c[0][0] == 0 ) and ( point_c[0][1] == 0 ) :
				point_c[0] = coor[1]
				countFull += 1
			elif ( point_c[1][0] == 0 ) and ( point_c[1][1] == 0 ):
				if distance(point_c[0], coor[1]) > 3000:
					point_c[1] = coor[1]
					countFull += 1
			else:
				pass
		
		print countFull
		
		PS = countForCoor
		
		if ( countFull == 2 ) & (not coor[0]):
			CaptureFlag = True
		
		
		if CaptureFlag:
			C_img = imTrim(frame, point_c)
			cv2.imshow('Capture',C_img)
			CaptureFlag = False
			point_c = [(0, 0), (0, 0)]
			countFull = 0
			
			r1=0

			img = cv2.imread('a2.jpg',0) #Rotating
			rows,cols = img.shape
			M = cv2.getRotationMatrix2D((cols/2,rows/2),r1,1) #rotate(r1)
			dst = cv2.warpAffine(img,M,(cols,rows))
			cv2.imwrite('sss.png',dst)
			speach =  pytesseract.image_to_string(Image.open('sss.png')).strip() 

			print speach
			tts = gTTS(text=speach, lang='en') #gTTS engine speach to english
			tts.save('b.mp3')
			os.system('mpg321 b.mp3') #playing the mp3file


		if not ret:
			print 'Not Found Devices'
			break
		
		
		''' show frames '''
		
		cv2.imshow('End Points 2', frame2)

		
		if cv2.waitKey(1) & 0xff == 27:
			break
			
	cap.release()
	cv2.destroyAllWindows()

def for_Testing2():
	os.system('sudo modprobe bcm2835-v4l2')
	cap = cv2.VideoCapture(0)
	#cap.set(3, 360)
	#cap.set(4, 240)
	num_c  = 0
	point_c = [(0, 0), (0, 0)]
	countFull = 0
	CaptureFlag = False
	PS = 0 #previous state
	pv = []
	''' Main Function'''
	theta = test_learn()
	ret, frame = cap.read() # 2592 * 1944

	start_line = "Now Loading...."
	tts = gTTS(text=start_line, lang='en')
	tts.save('greeting.mp3')
	os.system('mpg321 greeting.mp3')

	time.sleep(2)

	while True:
		ret, frame = cap.read() # 2592 * 1944
		frame2 = frame.copy() # copy frame 
		fr = frame.copy()
		
		'''Color Filter for finding hands '''
		
		hand_filter_down 	= np.array([0, 133, 77]) #YCrCb Filter low Limit
		hand_filter_up	= np.array([255, 173, 127]) #YCrCb Filter Up Limit
		
		hand_filter = handFilter(frame, hand_filter_up, hand_filter_down)
		hand = hand_filter.copy()
		
		st, con, hir = cv2.findContours(hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		coor2 = find_end2(con, fr, 5000, theta)
		
		print coor2
		
		countForCoor = len(coor2) - 1
		
		if (countForCoor == 1 ) & (PS == 0) & (coor2[0]):
			if ( point_c[0][0] == 0 ) and ( point_c[0][1] == 0 ) :
				point_c[0] = coor2[1]
				countFull += 1
			elif ( point_c[1][0] == 0 ) and ( point_c[1][1] == 0 ):
				if distance(point_c[0], coor2[1]) > 3000:
					point_c[1] = coor2[1]
					countFull += 1
			else:
				pass
		
		print countFull
		
		PS = countForCoor
		
		if ( countFull == 2 ) & (not coor2[0]):
			CaptureFlag = True
		
		
		if CaptureFlag:
			C_img = imTrim(frame, point_c)
			cv2.imshow('Capture',C_img)
			CaptureFlag = False
			point_c = [(0, 0), (0, 0)]
			countFull = 0
			r1=0

			img = cv2.imread('a33.png',0) #Rotating
			rows,cols = img.shape
			M = cv2.getRotationMatrix2D((cols/2,rows/2),r1,1) #rotate(r1)
			dst = cv2.warpAffine(img,M,(cols,rows))
			cv2.imwrite('sss.png',dst)
			speach =  pytesseract.image_to_string(Image.open('sss.png')).strip() 

			print speach
			tts = gTTS(text=speach, lang='en') #gTTS engine speach to english
			tts.save('b.mp3')
			os.system('mpg321 b.mp3') #playing the mp3file


		
		if not ret:
			print 'Not Found Devices'
			break
		
		
		''' show frames '''
		
		cv2.imshow('End Points 2', fr)

		
		if cv2.waitKey(1) & 0xff == 27:
			break
			
	cap.release()
	cv2.destroyAllWindows()
	
	
if __name__ == "__main__":
	for_Testing()
	#for_Testing2()
	pass