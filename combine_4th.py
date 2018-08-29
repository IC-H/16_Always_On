import cv2
import os
import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt


def imTrim(img, points):
	p1 = points[0]
	p2 = points[1]
	if p1[0] > p2[0]:
		x1 = p2[0]
		x2 = p1[0]
	else:
		x1 = p1[0]
		x2 = p2[0]
	if p1[1] > p2[1]:
		y1 = p2[1]
		y2 = p1[1]
	else:
		y1 = p1[1]
		y2 = p2[1]
	
	img_trim = img[x1:x2, y1:y2]
	cv2.imwrite('a2.png',img_trim)
	return img_trim

def handFilter(img, up, down):
	ker = np.ones((5,5), np.uint8) #filter of erode and dilate
	YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #YCrCb Transfer
	hand_filter = cv2.inRange(YCrCb, down, up) #Color filter
	return hand_filter


def findEndPoints(con, img, theta):
	px = 0
	py = 0
	count = 0
	coor = [ False ]
	
	n_c = len(con)
	for i in range(n_c):
		cnt = con[i]
		momen = cv2.moments(cnt)
		if momen['m00'] >100: # if area of closed contour is smaller than #, it may not be hand
			
			fea = ext_feat(momen)
		
			h = np.dot(fea, theta[1]) + theta[0][0]
	
			h = 1./(np.ones([1, 1]) + np.exp(-h))
			
			if h == 1.:
				coor[0] = True
				cx = int(momen['m10']/momen['m00']) # x-coordinate of center of mass
				cy = int(momen['m01']/momen['m00']) # y-coordinate of center of mass
				check = cv2.isContourConvex(cnt) # check whether it is convex or not
				# point out center of mass by blue circle 
				cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
				
				if  not check:
					hull = cv2.convexHull(cnt)
					cv2.drawContours(img, [hull], 0, (0, 255, 0), 3)
					
					phd = cv2.convexHull(cnt, returnPoints = False) 
					for j in range(len(phd)):
						const = phd[j]
						# current point of end-point
						crx = cnt[const][0][0][0]
						cry = cnt[const][0][0][1]
						absv = float(((crx - cx)**2 + (cry - cy)**2 )**(0.5)) 
					#If distance of  current points is in certain value then we difine that is End-Point.
						if  absv > 95 and absv < 140: 
							if (px - crx)**2 + (py - cry)**2 < 1600:
								count += 1
								px = int(( px*(count - 1) + crx)/count)
								py = int(( py*(count - 1) + cry )/count)
							else:
								if count == 0:
									px = crx
									py = cry
									count = 1
								elif count == 1:
									cv2.circle(img, (px, py), 5, (0, 0, 255), -1)
									coor.append((px, py))
									px = crx
									py = cry
								else:
									cv2.circle(img, (px, py), 5, (0, 0, 255), -1)
									coor.append((px, py))
									px = crx
									py = cry
									count = 1
					if (count != 0):
						cv2.circle(img, (px, py), 5, (0, 0, 255), -1)
						coor.append((px, py))

	return coor

def nom(v):
	v = np.array(v)
	vs = np.square(v)
	sv = np.sum(vs)
	av = math.sqrt(sv)
	nv = v/av
	return nv

def make_vector(p):
	n = len(p)
	nv = []
	for i in range(n):
		if i == n-1:
			v = p[0] - p[i]
			nv.append( nom(v) )
		else:
			v = p[i+1] - p[i]
			nv.append( nom(v) )
	return nv

def update_point(p, img):
	dist = 3000
	n = len(p)
	np = []
	pp = 0
	for i in range(n):
		if i == n-1:
			if distance(p[0], p[pp]) > dist:
				cv2.circle(img, (p[pp][0], p[pp][1]), 2, (255, 0, 0), -1)
				np.append( p[pp] )
		else:
			if distance(p[i + 1], p[pp] ) > dist:
				pp = i
				cv2.circle(img, (p[pp][0], p[pp][1]), 2, (255, 0, 0), -1)
				np.append( p[pp] )
	return np

def make_vector_debug(p, img):
	n = len(p)
	nv = []
	pp = 0
	for i in range(n):
		if i == n-1:
			v = p[0] - p[i]
			cv2.circle(img, (p[i][0], p[i][1]), 1, (255, 0, 0), -1)
			nv.append( nom(v) )
		else:
			v = p[i+1] - p[i]
			cv2.circle(img, (p[i][0], p[i][1]), 1, (255, 0, 0), -1)
			nv.append( nom(v) )
	return nv


def cal_deg(v1, v2):
	av1 = math.sqrt( v1[0]**2 + v1[1]**2 )
	av2 = math.sqrt( v2[0]**2 + v2[1]**2 )
	c = ( v1[0]*v2[0] + v1[1]*v2[1] )/(av1*av2)
	if c >= 1.0:
		return 0
	elif c <= -1:
		return 180
	else:
		theta = math.acos(c)*180/math.pi
		return theta
	
def make_deg(v):
	n = len(v)
	deg = []
	for i in range(n):
		if i == 0:
			deg.append(cal_deg(v[n-1], v[0]))
		else:
			deg.append(cal_deg(v[i], v[i - 1]))
	return deg

def find_end2(con, img, m00, theta):
	#img2 = img.copy()
	n_c = len(con)
	coor = [False]
	
	for i in range(n_c):
		cnt = con[i]
		momen = cv2.moments(cnt)
		if momen['m00'] >m00: # if area of closed contour is smaller than #, it may not be hand
			
			
			fea = ext_feat(momen)
		
			h = np.dot(fea, theta[1]) + theta[0][0]
	
			h = 1./(np.ones([1, 1]) + np.exp(-h))
			
			p = []
			n = len(cnt)
			c_d = []
			
			if h == 1.0:
				coor[0] = True
			
			for j in range(n):
				p.append(cnt[j][0])
			n_p = update_point(p, img)
			
			#cv2.namedWindow('debug2', cv2.WINDOW_NORMAL)
			#cv2.imshow('debug2', img)
			
			nv = make_vector(n_p)
			
			deg = make_deg(nv)
			
			for j in range(len(nv)):
				if  deg[j] > 120:
					c_d.append(j)
					cv2.circle(img, (n_p[j][0], n_p[j][1]), 5, (0, 0, 255), -1)
					coor.append( n_p[j] )
			
			
	return coor

def handle_video():
	os.system('sudo modprobe bcm2835-v4l2')
	# Capture Video 
	cap = cv2.VideoCapture(0)
	cap.set(3, 360) #set width of cap size
	cap.set(4, 240) #set height of cap size
	num_c  = 0
	point_c = [(0, 0), (0, 0)]
	countFull = 0
	CaptureFlag = False
	PS = 0
	pv = []
	''' Main Function'''
	theta = test_learn()

	while True:
		ret, frame = cap.read() # 2592 * 1944
		frame2 = frame.copy() # copy frame 
		fr = frame.copy()
		
		'''Color Filter for finding hands '''
		
		hand_filter_down 	= np.array([0, 133, 77]) #YCrCb Filter low Limit
		hand_filter_up	= np.array([255, 173, 127]) #YCrCb Filter Up Limit
		
		hand_filter = handFilter(frame, hand_filter_up, hand_filter_down)
		hand = hand_filter.copy()
		
		con, hir = cv2.findContours(hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		coor = findEndPoints(con, frame2, theta)
		
		coor2 = find_end2(con, fr, 5000, theta)
		
		print coor2
		
		countForCoor = len(coor2) - 1
		
		if (countForCoor == 1 ) & (PS == 0):
			if ( point_c[0][0] == 0 ) and ( point_c[0][1] == 0 ) :
				point_c[0] = coor2[1]
				countFull += 1
			elif ( point_c[1][0] == 0 ) and ( point_c[1][1] == 0 ):
				if distance(point_c[0], coor2[1]) > 30000:
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
			#break
		
		if not ret:
			print 'Not Found Devices'
			break
		
		
		''' show frames '''
		
		cv2.imshow('Frame', frame)
		cv2.imshow('Color Filter', hand_filter)
		cv2.drawContours(frame, con, -1, (0, 255, 0), 1)
		cv2.imshow('contour',frame)
		#cv2.imshow('End Points 1',frame2)
		cv2.imshow('End Points 2', fr)

		
		if cv2.waitKey(1) & 0xff == 27:
			break
			
	cap.release()
	cv2.destroyAllWindows()

def makeString(img, contour, mon):
	shp = img.shape
	b_img = np.zeros((shp[0],  shp[1], 3), np.uint8)
	size_of_img = shp[0]*shp[1]
	cv2.drawContours(b_img, contour, -1, (0, 255, 0), 1)
	cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
	cv2.imshow('contour', b_img)
	Flag_Hand = cv2.waitKey(0)
	if Flag_Hand & 0xff == 27:
		return 27
	cv2.destroyWindow('contour')
	s = chr(Flag_Hand) + '\t'
	s += str(mon['m00']/size_of_img)+ '\t'
	s += str(mon['m10'])+ '\t'
	s += str(mon['m01'])+ '\t'
	s += str(mon['m20'])+ '\t'
	s += str(mon['m11'])+ '\t'
	s += str(mon['m02'])+ '\t'
	s += str(mon['m30'])+ '\t'
	s += str(mon['m21'])+ '\t'
	s += str(mon['m12'])+ '\t'
	s += str(mon['m03'])+ '\t'
	
	s += str(mon['mu20'])+ '\t'
	s += str(mon['mu11'])+ '\t'
	s += str(mon['mu02'])+ '\t'
	s += str(mon['mu30'])+ '\t'
	s += str(mon['mu21'])+ '\t'
	s += str(mon['mu12'])+ '\t'
	s += str(mon['mu03'])+ '\t'
	
	s += str(mon['nu20'])+ '\t'
	s += str(mon['nu11'])+ '\t'
	s += str(mon['nu02'])+ '\t'
	s += str(mon['nu30'])+ '\t'
	s += str(mon['nu21'])+ '\t'
	s += str(mon['nu03'])+ '\n'
	
	return s

def make_data():
	
	os.system('sudo modprobe bcm2835-v4l2')
	# Capture Video 
	cap = cv2.VideoCapture(0)
	cap.set(3, 360) #set width of cap size
	cap.set(4, 240) #set height of cap size
	
	#data = sp.genfromtxt('data_test.tsv', delimiter = '\t')
	#print len(data)
	
	while True:
		ret, frame = cap.read()
		Ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
		hand_filter_down 	= np.array([0, 133, 77]) #YCrCb Filter low Limit
		hand_filter_up	= np.array([255, 173, 127]) #YCrCb Filter Up Limit
		
		hand_filter = handFilter(frame, hand_filter_up, hand_filter_down)
		
		hand = hand_filter.copy()

		#cv2.drawContours(b_img, con, -1, (0, 255, 0), 1)
		cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
		#cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
		cv2.imshow('origin', frame)
		#cv2.imshow('contour', b_img)
		
		ki = cv2.waitKey(1) & 0xff
		if ki == 27:
			break
		elif ki == ord('s'):
			con, hir = cv2.findContours(hand_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			n_c = len(con)
			f_d = open('data_test.tsv', 'a+')
			
			for i in range(n_c):
				cnt = con[i]
				mon = cv2.moments(cnt)
				if mon['m00'] > 1000:
					s = makeString(frame, cnt, mon)
					if s == 27:
						break
					f_d.write(s)
			
			f_d.close()
			cv2.destroyAllWindows()
			
		cv2.namedWindow('Color Filter', cv2.WINDOW_NORMAL)
		cv2.imshow('Color Filter', hand)

		
	
	cap.release()
	cv2.destroyAllWindows()
	
def distance(p0, p1):
	p0 = np.array(p0)
	p1 = np.array(p1)
	return np.sum((p0 - p1)**2)

def nn_classify(training_set, training_labels, new_ex):
	dists = np.array([distance(t, new_ex) for t in training_set])
	nearest = dists.argmin()
	return training_labels(nearest)

def t_linear_regression(x, y, r, l):
	x = np.array(x)
	y = np.array([y], np.uint8)
	y = y.T
	#y1 = y & 0b001
	y2 = (y & 0b010)/2

	#y3 = (y & 0b100)/4
	
	#y2 = np.array(y2, np.float64)
	
	x_s = x.shape
	m = x_s[0]
	n = x_s[1]
	theta = np.zeros([n, 1])
	b = np.zeros([m, 1])
	a = 0.01
	h = np.dot(x, theta) + b
	
	h = 1./(np.ones([m, 1]) + np.exp(-h))
	
	for i in range(r):
		d = 1./m*np.dot(x.T, (h - y2))
		
		b = b - a*(h-y2)
		
		theta = theta - a*d + l/m*theta
		
		h = np.dot(x, theta) + b
		
		h = 1./(np.ones([m, 1]) + np.exp(-h))
	
	return [b, theta]

def ext_feat(mom):
	return np.array([[mom['m00'], mom['m10'], mom['m01'], mom['m20'], mom['m11'], mom['m02'],\
	mom['m30'], mom['m21'], mom['m12'], mom['m03'], mom['mu20'], mom['mu11'], mom['mu02'],\
	mom['mu30'], mom['mu21'], mom['mu12'], mom['mu03'], mom['nu20'], mom['nu11'], mom['nu02'],\
	mom['nu30'], mom['nu21'], mom['nu03']]])

def test_learn():
	
	raw_data = sp.genfromtxt('data_test.tsv', delimiter = '\t')
	
	print len(raw_data)
	
	d_s = raw_data.shape
	
	m = d_s[0]
	
	n = d_s[1]
	
	hf = raw_data[:m-1, 0]
	hf = np.array(hf, np.uint8)
	hf2 = (hf & 0b10)/2
	
	
	t_d = raw_data[:m-1, 1:]

	e_d = np.array([raw_data[m-1, 1:]])
		
	#print e_d.shape

	theta = t_linear_regression(t_d, hf, 1000, 107)
	
	h = np.dot(e_d, theta[1]) + theta[0][0]
	
	#print h.shape
	
	h = 1./(np.ones([1, 1]) + np.exp(-h))
	
	return theta
	
	#print h
	
	#print raw_data[m-1, 0]

def test_tt():
	
	theta = test_learn()
	
	
	count = 0.
	fcount = 0.
	m00 = 3000
	img = cv2.imread('a5.jpg')
	frame = img.copy()
	frame2 = img.copy()
	frame3 = img.copy()
	hand_filter_down = np.array([0, 133, 77]) #YCrCb Filter low Limit
	hand_filter_up	= np.array([255, 173, 127]) #YCrCb Filter Up Limit
	
	hand_filter = handFilter(img, hand_filter_up, hand_filter_down)
	hand = hand_filter.copy()
	hand2 = hand.copy()
	
	con, hir = cv2.findContours(hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	n_c = len(con)
	
	for i in range(n_c):
		
		shp = img.shape
		bg = np.zeros((shp[0],  shp[1], 3), np.uint8)
		mon = cv2.moments(con[i])
		
		fea = ext_feat(mon)
		
		h = np.dot(fea, theta[1]) + theta[0][0]
	
		h = 1./(np.ones([1, 1]) + np.exp(-h))
		
		print h
		
		if h == 1.0:
			cv2.drawContours(bg, con[i], -1, (0, 255, 0), 1)
			cv2.namedWindow('hand?', cv2.WINDOW_NORMAL)
			cv2.imshow('hand?', bg)
		else:
			cv2.drawContours(bg, con[i], -1, (0, 0, 255), 1)
			cv2.namedWindow('not hand?', cv2.WINDOW_NORMAL)
			cv2.imshow('not hand?', bg)

		key = cv2.waitKey(0)
		cv2.destroyAllWindows()
		if key == ord('y'):
			count += 1.
		elif key == 27:
			break
		else:
			fcount += 1
			pass
		
	print count/(fcount + count)

	key = cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	
	

def test_and_update():
	
	theta = test_learn()
	
	
	count = 0.
	fcount = 0.
	m00 = 3000
	img = cv2.imread('a1.jpg')
	frame = img.copy()
	frame2 = img.copy()
	frame3 = img.copy()
	hand_filter_down = np.array([0, 133, 77]) #YCrCb Filter low Limit
	hand_filter_up	= np.array([255, 173, 127]) #YCrCb Filter Up Limit
	
	hand_filter = handFilter(img, hand_filter_up, hand_filter_down)
	hand = hand_filter.copy()
	hand2 = hand.copy()
	
	con, hir = cv2.findContours(hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	#coor = findEndPoints(con, frame2)
	
	#coor2 = find_end2(con, frame, m00)
	
	n_c = len(con)
	
	for i in range(n_c):
		
		shp = img.shape
		bg = np.zeros((shp[0],  shp[1], 3), np.uint8)
		mon = cv2.moments(con[i])
		
		fea = ext_feat(mon)
		
		h = np.dot(fea, theta[1]) + theta[0][0]
	
		h = 1./(np.ones([1, 1]) + np.exp(-h))
		
		print h
		
		if h == 1.0:
			cv2.drawContours(bg, con[i], -1, (0, 255, 0), 1)
			cv2.namedWindow('hand?', cv2.WINDOW_NORMAL)
			cv2.imshow('hand?', bg)
		else:
			cv2.drawContours(bg, con[i], -1, (0, 0, 255), 1)
			cv2.namedWindow('not hand?', cv2.WINDOW_NORMAL)
			cv2.imshow('not hand?', bg)

		key = cv2.waitKey(0)
		cv2.destroyAllWindows()
		if key == ord('y'):
			count += 1.
		elif key == 27:
			break
		else:
			fcount += 1
			pass
	
	print count/(count + fcount)

def Neral_Network():
	pass

if __name__ == '__main__':
	#make_data()
	handle_video()
	#test_learn()
	#test_tt()
	#test_and_update()

