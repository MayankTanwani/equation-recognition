import cv2
import numpy as np
import sys, getopt
from skimage.feature import hog
from skimage import data, exposure
from keras.models import load_model
from symbols import *
from classification import *
from make_mst import *
import easygui

def main(argv):
	inp_pic = "timages/"
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:  
		print ('test.py -i <inputfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('test.py -i <inputfile>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inp_pic += arg

	clf = Classifiy()
	org = cv2.imread(inp_pic)
	cv2.imshow("Original Image", org)
	img = cv2.imread(inp_pic,0)

	img = cv2.medianBlur(img,3)
	img = cv2.GaussianBlur(img, (5, 5), 0)
	cv2.imshow("After Blur",img)

	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
	kernel = np.ones((1,1),np.uint8)

	# Dilation or Thinning of Image
	img = cv2.dilate(img,kernel, iterations = 4)
	
	cv2.imshow("After", img)

	thresh = img
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]

	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle
	print("Deskew Angle -> {}".format(angle))

	(h, w) = img.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(img, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	org = cv2.warpAffine(org, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	img = rotated

	cv2.imshow("Deskew",img)

	_,ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	new_ctrs = []

	for i in range(0, len(ctrs)):
		length = cv2.contourArea(ctrs[i])
		print(length)
		if length>20:
			new_ctrs.append(ctrs[i])

	cv2.imshow("Corrected Contours",img)
	rects = [cv2.boundingRect(ctr) for ctr in new_ctrs]
	all_contours = []
	recognised_symbols = []
	imageWithSymbols = org.copy()
	rects = sorted(rects,key = lambda x : x[0])
	final_equation = ""
	for rect in rects:
		# Draw the rectangles
		cv2.rectangle(org, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 212, 212), 3) 
		cv2.rectangle(imageWithSymbols, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 

		leng = int(rect[3] * 1.4)

		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		roi = img[pt1:pt1+leng, pt2:pt2+leng]
		roi_color = org[pt1:pt1+leng, pt2:pt2+leng]
		roi = cv2.resize(roi, (49, 49), interpolation=cv2.INTER_AREA)
		all_contours.append([roi,pt1+leng/2.0,pt2 + leng/2.0])
		symbol = clf.recognise_symbol(roi)
		# symbol = clf.recognise_cnn_svm(roi)
		recognised_symbols.append(str(symbol))
		cv2.putText(imageWithSymbols, str(symbol), (int(rect[0]), int(rect[1])),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
		print(str(symbol))
	cv2.imshow("Resulting Image with Predicted numbers", imageWithSymbols)
	cv2.waitKey()

	result = spanning_tree(org,img,all_contours)
	final_array = make_eqaution(result,all_contours,recognised_symbols)
	for i in final_array:
		final_equation = final_equation + str(i) + " "
	easygui.msgbox(final_equation, 'Recognised Equation')

if __name__ == "__main__":
	main(sys.argv[1:])