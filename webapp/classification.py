import cv2
import numpy as np
import sys, getopt
from skimage.feature import hog
from skimage import data, exposure
from keras.models import load_model
from symbols import *
from connected_componects import *
import gist
import pickle

class Classifiy:
	def __init__(self):
		self.model = load_model('hog-crohme-digits-20e-5-features.h5')

	def recognise_symbol(self,roi):
		fd,hog_image = hog(roi,pixels_per_cell = (5,5),visualize=True)
		fd = fd[np.newaxis,:]
		nbr = self.model.predict_classes(fd,verbose=0)
		reg_class = nbr[0]
		return decode_sym(reg_class)

	def recognise_cnn_svm(self,roi):
		model = load_model('crohme-half-model-features--cnn-20e.h5')
		clf = pickle.load(open("svm_model_features-new-20e.sav","rb"))
		features = model.predict(roi[np.newaxis,:,:,np.newaxis])
		nbr = clf.predict(features)
		reg_class = nbr[0]
		return decode_sym(reg_class)

	def recognise_gist(self,roi):
		clf = pickle.load(open("svm_model_gist.sav","rb"))
		fd = gist.extract(roi)
		fd = fd[np.newaxis,:]
		nbr = clf.predict(fd)
		reg_class = nbr[0]
		return decode_sym(reg_class)

	def recognise_hog_svm(self,roi):
		clf = pickle.load(open("finalized_model_HOG_4x4.sav","rb"))
		fd = hog(roi,pixels_per_cell=(4,4))
		fd = fd[np.newaxis,:]
		nbr = clf.predict(fd)
		reg_class = nbr[0]
		return decode_sym(reg_class)
