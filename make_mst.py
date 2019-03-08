import cv2
import matplotlib.pyplot as plt
from mst import *
import numpy as np
import sys, getopt
from skimage.feature import hog
import queue

def spanning_tree(org,img,all_contours):
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
	cv2.waitKey()
	return result

def make_eqaution(mst,all_contours,recognised_symbols):
	all_dict = {}
	for i in range(len(recognised_symbols)):
		all_dict[i] = ['a',[]]
		# print(all_dict.get(i)[1])
	for u,v,w in mst : 
		cur_list = all_dict.get(u)[1]
		cur_list.append(v)
		all_dict[u] = [recognised_symbols[u],cur_list]
	print("Dictionary : : ")
	print(all_dict)
	return bfs(all_dict,0,recognised_symbols)

def bfs(result,src,recognised_symbols) :
	q = queue.Queue()
	q.put(src)
	visited = [False] * len(recognised_symbols)
	visited[src] = True
	traversal = []
	while not q.empty():
		node = q.get()
		traversal.append(recognised_symbols[node])
		for v in result.get(node)[1]:
			if not visited[v] :
				q.put(v)
				visited[v] = True
	# print (traversal)
	return correct_eqaution(traversal)

def correct_eqaution(traversal):
	final_eqaution = ""
	new_traversal = []
	i=0
	while i != (len(traversal)-1) :
		if traversal[i] == traversal[i+1] == '-':
			new_traversal.append("=")
			i+=2
			continue
		new_traversal.append(traversal[i])
		i+=1
	new_traversal.append(traversal[-1]) 
	print(new_traversal)
	return new_traversal

