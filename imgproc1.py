import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops


img = cv2.imread(r'C:\Users\ASUS\Desktop\Minor_Project\images\gt\0_50.png',cv2.IMREAD_COLOR)
#img = cv2.imread(r'C:\Users\ASUS\Desktop\1_73.png',cv2.IMREAD_COLOR)

#img = cv2.imread(r'C:\Users\ASUS\Desktop\Minor_Project\final_dataset\75.png',cv2.IMREAD_COLOR)

blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow('Gaussian Filter',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image',img)
px = img[55,55]
print(px)
cv2.waitKey(0)
cv2.destroyAllWindows()

frame = img
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

cv2.imshow('frame', frame)
cv2.imshow('hsv', hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2gray = cv2.cvtColor(hsv,cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(img2gray, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed",closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

gradient = cv2.morphologyEx(img2gray, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("Gradient",gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

open = cv2.morphologyEx(img2gray, cv2.MORPH_OPEN, kernel)
cv2.imshow("Opening",open)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret,thresh1 = cv2.threshold(closing,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(closing,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(closing,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(closing,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(closing,127,255,cv2.THRESH_TOZERO_INV)

cv2.imshow('Thresh', thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]


for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()


#thresh4 = cv2.cvtColor(thresh4,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(thresh4,100,200)
cv2.imshow("edges",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

sobelx = cv2.Sobel(thresh4,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(thresh4,cv2.CV_64F,0,1,ksize=5)  # y
cv2.imshow("Sobelx",sobelx.astype('uint8')*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Sobely",sobely.astype('uint8')*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

sobelx=sobely.astype('uint8')*255
sobely=sobely.astype('uint8')*255
sobel=sobelx+sobely
cv2.imshow("Sobel",sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_arr= np.array(sobel)
#img_arr = img_arr[:,:,0]
print(img_arr.shape)
feat_lbp = local_binary_pattern(img_arr,8,1,'uniform') #Radius = 1, No. of neighbours = 8
feat_lbp = np.uint8((feat_lbp/feat_lbp.max())*255) #Converting to unit8
lbp_img = PIL.Image.fromarray(feat_lbp) #Conversion from array to PIL image
plt.imshow(lbp_img,cmap='gray') #Displaying LBP
lbp_arr = np.array(lbp_img)
cv2.imshow("LBP",lbp_arr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Energy and Entropy of LBP feature
lbp_hist,_ = np.histogram(feat_lbp,8)
lbp_hist = np.array(lbp_hist,dtype=float)
lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
lbp_energy = np.sum(lbp_prob**2)
lbp_entropy = -np.sum(np.multiply(lbp_prob,np.log2(lbp_prob)))
print('LBP energy = '+str(lbp_energy))
print('LBP entropy = '+str(lbp_entropy))

# Finding GLCM features from co-occurance matrix
gCoMat = greycomatrix(img_arr, [2], [0],256,symmetric=True, normed=True) # Co-occurance matrix
contrast = greycoprops(gCoMat, prop='contrast')
dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
homogeneity = greycoprops(gCoMat, prop='homogeneity')
energy = greycoprops(gCoMat, prop='energy')
correlation = greycoprops(gCoMat, prop='correlation')
n_white_pix = np.sum(thresh1 == 255)
print('Contrast = '+str(contrast[0][0]))
print('Dissimilarity = '+str(dissimilarity[0][0]))
print('Homogeneity = '+str(homogeneity[0][0]))
print('Energy = '+str(energy[0][0]))
print('Correlation = '+str(correlation[0][0]))
print("Area = ",n_white_pix)



