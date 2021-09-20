# Import the modules
import cv2
import numpy as np
import sys, getopt
from skimage.feature import hog
from skimage import data, exposure
from keras.models import load_model
from symbols import *

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


    # Load the Keras CNN trained model
    model = load_model('tmodels/hog-crohme-digits-10e-4,2-features.h5')

    # Original image
    im = cv2.imread(inp_pic)
    cv2.imshow("Original Image", im)
    cv2.waitKey()



    # Read image in grayscale mode
    img = cv2.imread(inp_pic,0)
    
    # Median Blur and Gaussian Blur to remove Noise
    img = cv2.medianBlur(img,3)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive Threshold for handling lightning
    im_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    
    kernel = np.ones((1,1),np.uint8)

    # Dilation or Thinning of Image
    im_th = cv2.dilate(im_th,kernel, iterations = 4)
    
    cv2.imshow("After", im_th)
    


# cv2.imshow("Threshold Image",im_th)

    # Find contours in the image
    _,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, predict using cnn model
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (49, 49), interpolation=cv2.INTER_AREA)

        fd,hog_image = hog(roi,pixels_per_cell=(4,4),cells_per_block=(2,2),visualize=True)

        fd = fd[np.newaxis,:]

        # Input for CNN Model
        # roi = roi[np.newaxis,:,:,np.newaxis]

        # Input for Feed Forward Model
        # roi = roi.flatten()
        # roi = roi[np.newaxis]
        nbr = model.predict_classes(fd,verbose=0)
        cv2.putText(im, str(decode_sym(nbr[0])), (int(rect[0]), int(rect[1])),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        print(str(int(nbr[0])))
    cv2.imshow("Resulting Image with Predicted numbers", im)
    cv2.waitKey()

if __name__ == "__main__":
   main(sys.argv[1:])