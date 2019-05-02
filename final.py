import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import PIL
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

image_path = r'C:\Users\ASUS\Desktop\Minor_Project\final_dataset'

################################### Loading the images from the path ###################################
def loadImages(path):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([os.path.join(path, file)
                          for file in os.listdir(path) if file.endswith('.png')])
    return image_files


imgfiles = loadImages(image_path)

#######################################################################################################

############################ Segmentation of the Tumor using Morphology and Thresholding #################
def segmentation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img2gray, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(img2gray, cv2.MORPH_GRADIENT, kernel)
    open = cv2.morphologyEx(img2gray, cv2.MORPH_OPEN, kernel)
    ret, thresh1 = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(closing, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    # thresh4 = cv2.cvtColor(thresh4, cv2.COLOR_BGR2GRAY)
    return thresh1

########################################################################################################

################################ Saving the segmented images #############################################

def directory_traversal():
    location = r"C:\Users\ASUS\Desktop\Minor_Project\final_dataset"
    seg_list = []
    for i in os.listdir(location):
        if i.endswith(".png"):
            seg_list.append(os.path.join(location, i))
    c = 0
    new_location = r"C:\Users\ASUS\Desktop\Minor_Project\seg_imgs"
    for i in seg_list:
        if i.endswith(".png"):
            img = cv2.imread(i, cv2.IMREAD_COLOR)
            img_arr = segmentation(img)
            seg_img = PIL.Image.fromarray(img_arr)
            seg_location = new_location + '\seg_img_' + str(c) + '.png'
            c = c + 1
            seg_img.save(seg_location)


directory_traversal()

###########################################################################################################


##########################  Feature Extraction ############################################
ener = []
diss = []


def feature_extraction(sobel):
    img_arr = np.array(sobel)
    # img_arr = img_arr[:,:,0]
    #print(img_arr.shape)
    feat_lbp = local_binary_pattern(img_arr, 8, 1, 'uniform')  # Radius = 1, No. of neighbours = 8
    feat_lbp = np.uint8((feat_lbp / feat_lbp.max()) * 255)  # Converting to unit8
    lbp_img = PIL.Image.fromarray(feat_lbp)  # Conversion from array to PIL image
    plt.imshow(lbp_img, cmap='gray')  # Displaying LBP
    lbp_arr = np.array(lbp_img)

    # Energy and Entropy of LBP feature
    lbp_hist, _ = np.histogram(feat_lbp, 8)
    lbp_hist = np.array(lbp_hist, dtype=float)
    lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
    lbp_energy = np.sum(lbp_prob ** 2)
    lbp_entropy = -np.sum(np.multiply(lbp_prob, np.log2(lbp_prob)))

    # Finding GLCM features from co-occurance matrix
    gCoMat = greycomatrix(img_arr, [2], [0], 256, symmetric=True, normed=True)  # Co-occurance matrix
    contrast = greycoprops(gCoMat, prop='contrast')
    dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
    homogeneity = greycoprops(gCoMat, prop='homogeneity')
    energy = greycoprops(gCoMat, prop='energy')
    correlation = greycoprops(gCoMat, prop='correlation')
    ener.append(float(energy[0][0]) * 100)
    diss.append(float(dissimilarity[0][0]) * 1000)

###########################################################################################################

################################### Edge Detection ########################################################

def edge_detection():
    location = r"C:\Users\ASUS\Desktop\Minor_Project\seg_imgs"
    seg_list = []
    for i in os.listdir(location):
        if i.endswith(".png"):
            seg_list.append(os.path.join(location, i))
    c = 0
    for i in seg_list:
        if i.endswith(".png"):
            img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
            edges = cv2.Canny(img, 100, 200)
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y
            sobelx = sobely.astype('uint8') * 255
            sobely = sobely.astype('uint8') * 255
            sobel = sobelx + sobely
            feature_extraction(sobel)


edge_detection()
print(ener)
print(diss)

############################################################################################################

################################# Finding area of the Tumpor ###############################################

def pixels():
    location = r"C:\Users\ASUS\Desktop\Minor_Project\seg_imgs"
    seg_list = []
    for i in os.listdir(location):
        if i.endswith(".png"):
            seg_list.append(os.path.join(location, i))
    c = 0
    pix = []
    for i in seg_list:
        if i.endswith(".png"):
            img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
            n_white_pix = np.sum(img == 255)
            pix.append(n_white_pix)
            # print('Number of white pixels:', n_white_pix)
    print(pix)
    return pix


pixs = pixels()
mean = sum(pixs) / len(pixs)
print(mean)

####################################################################################################################

test_diss = [3999.471830985916, 3848.767605633803, 2078.1690140845067, 1852.2887323943658, 1759.1549295774646, 2067.957746478873, 2323.239436619718, 2084.330985915493, 2274.471830985916, 2125.176056338028, 2170.774647887324, 2176.4084507042253, 4204.929577464789, 2105.9859154929572, 2014.260563380282, 2082.394366197183, 2133.450704225352, 2316.021126760564, 3258.6267605633802, 3687.1478873239435, 3563.0281690140846, 3763.0281690140846, 3254.049295774648, 3825.5281690140846, 3723.2394366197186, 4202.288732394367, 4003.1690140845076, 4345.070422535212, 4688.380281690142, 4263.908450704225]
test_area = [1570, 2020, 389, 476, 627, 747, 837, 915, 973, 1036, 1090, 1117, 2485, 1143, 1182, 1234, 1264, 1326, 3465, 3313, 2873, 2505, 2158, 2749, 2999, 3284, 3538, 3719, 3802, 3914]
X_test = np.column_stack((test_diss, test_area))
print(X_test)


################################### Classifiction using K-Means Clustering ########################################

X = np.column_stack((diss, pixs))
print(X)

plt.scatter(X[:,0], X[:,1], label='True Position')
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.xlabel("Dissimilarity")
plt.ylabel("Size")
plt.title("Classification of Tumor as HGG or LGG")
plt.legend(loc=2)
plt.show()

##################################################################################################################

################################## Test Set Prediction and Accuracy Calculation ##################################

y_pred = kmeans.predict(X_test)
print(y_pred)
y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
print(len(y_true))
l = len(y_true)

acc = accuracy_score(y_true, y_pred,normalize=False)
print(acc)

acc_perc = (acc/l)*100;
print("Accuracy :", acc_perc)

#################################################################################################################

