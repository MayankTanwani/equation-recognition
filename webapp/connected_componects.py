import cv2
import matplotlib.pyplot as plt
from mst import *
import numpy as np
import sys, getopt
from skimage.feature import hog
from skimage import data, exposure


def spanning_tree(org,img,ctrs):

	def get_Rectangles(newRectangles,newImage):
		copyImage = newImage.copy()
		index = 0
		all_contours = []
		for rect in newRectangles:
			leng_y = int(rect[3] * 1)
			leng_x = int(rect[2] * 1.4)
			pt1 = int(rect[1] + rect[3] // 2 - leng_y // 2)
			pt2 = int(rect[0] + rect[2] // 2 - leng_x // 2)
			roi = copyImage[pt1:pt1+leng_y, pt2:pt2+leng_x]
			nbr = "1"
			index += 1
			# cv2.imshow("ctr_org"+str(rects.index(rect)),roi)
			all_contours.append([roi,pt1+leng_y/2.0,pt2 + leng_x/2.0])
			cv2.rectangle(org, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 212, 212), 3)
			cv2.putText(newImage, str((index)), (int(rect[0]), int(rect[1])),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 212, 25), 3)
		return all_contours

	# Get rectangles contains each contour
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]


	all_contours = get_Rectangles(rects,img)


	all_contours = sorted(all_contours,key = lambda x : x[2])


	g = Graph(len(all_contours))
	for i in range(len(all_contours)):
		for j in range(len(all_contours)):
			if j < i : 
				continue
			g.addEdge(i,j,calculate_weight(all_contours[i][1:],all_contours[j][1:]))

	result = g.KruskalsMST()
	combined_result = []
	i = 0
	for cont,x,y in all_contours : 
		print("{}: x = {},y= {}".format(i,y,x))
		i+=1
		cv2.rectangle(org,(int(y)-2,int(x)-2),(int(y+2),int(x+2)),(0,0,255),5)
	print("\nMST for the image\n")
	printMST(result)
	# cv2.imshow("Countours",org)
	for u,v,w in result : 
		combined_result.append([u,all_contours[u],v,all_contours[v],w])
		# cv2.imshow("{} -connected with- {} ".format(u,v),all_contours[u][0])	
		cv2.line(org,(int(all_contours[u][2]),int(all_contours[u][1])),(int(all_contours[v][2]),int(all_contours[v][1])),(255,0,0),2)
	# cv2.imshow("MST",org)
	# cv2.waitKey()

if __name__ == "__main__":
   main(sys.argv[1:])