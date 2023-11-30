from scipy.spatial.distance import directed_hausdorff as haus_dist
from scipy import signal
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob, os
import math

# def cart2pol(x, y):
#     rho = np.sqrt(x**2 + y**2)
#     phi = np.arctan2(y, x)
#     return(rho, phi)

# def pol2cart(rho, phi):
#     x = rho * np.cos(phi)
#     y = rho * np.sin(phi)
#     return(x, y)

img = cv.imread("JigsawTrainingData\MjU-padding.bmp")
ht,wd = img.shape[:2]

# add padding to img
old_image_height, old_image_width, channels = img.shape

new_img_w = old_image_width + 10
new_img_h = old_image_height + 10

result = np.full((new_img_h,new_img_w,channels),(0,0,0),dtype=np.uint8)

x_center = (new_img_w - img.shape[0]) // 2
y_center = (new_img_h - img.shape[1]) // 2

result[y_center:y_center+img.shape[0], 
       x_center:x_center+img.shape[1]] = img

img = result

img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # preprocessing
# kernel_size = 3
# kern = np.ones((3,3))


# blur_gray = cv.GaussianBlur(img_gray,(kernel_size,kernel_size),0)

# thresh = cv.threshold(blur_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

# contours, _ = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

# m = cv.moments(contours[0])
# cx = int(m["m10"]/m["m00"])
# cy = int(m["m01"]/m["m00"])

# polar = [cart2pol(c[0][0] - cx, c[0][1] - cy) for c in contours[0][:]]
# max_i = polar.index(max(polar, key = lambda x: x[1]))
# polar = polar[max_i:] + polar[:max_i]
# ds, phis = zip(*polar)

# ds = signal.medfilt(ds,kernel_size)
# ds_deriv = np.gradient(ds)
# b,a = ([1/5,1/5,1/5,1/5,1/5],[1])
# ds_deriv2 = abs(signal.filtfilt(b,a,ds_deriv))
# peaks = signal.find_peaks(ds_deriv2,distance=2)
# corners_x = []
# corners_y = []
# for i in peaks[0]:
#     phi = phis[i]
#     rho = ds[i]
#     x,y = pol2cart(rho,phi)
#     corners_x.append(x+new_img_w/2)
#     corners_y.append(y+new_img_h/2)

# plt.figure(1)
# plt.plot(phis,ds_deriv)
# plt.title('raw polar')
# plt.figure(2)
# plt.plot(phis,ds_deriv2)
# plt.title('filtered polar')
# plt.figure(3)
# plt.imshow(thresh)
# plt.scatter(corners_x,corners_y)
# plt.figure(4)
# plt.plot(phis,ds)
# plt.title('raw-er polar')
# plt.show()

class getSeparatedEdges():

    def get_edges(img):

        # add padding to img
        old_image_height, old_image_width, channels = img.shape
        new_img_w = old_image_width + 10
        new_img_h = old_image_height + 10
        result = np.full((new_img_h,new_img_w,channels),(0,0,0),dtype=np.uint8)
        x_center = (new_img_w - img.shape[0]) // 2
        y_center = (new_img_h - img.shape[1]) // 2

        result[y_center:y_center+img.shape[0], 
            x_center:x_center+img.shape[1]] = img
        img = result
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        dst = cv.cornerHarris(img_gray,2,3,.04)
        dst = cv.dilate(dst,None)
        img[dst>.08*dst.max()] = [0,0,255]

        _, dst = cv.threshold(dst,0.08*dst.max(),255,0)
        dst = np.uint8(dst)
        _, _, _, centroids = cv.connectedComponentsWithStats(dst)

        centroids = centroids[1:]
        distances = []

        #find 4 corners that are most remote (rotation invariant)

        #use cv.warpAffine later to guarantee straight lines? rotate
        for i in centroids:
            smallest = 300
            for j in range(len(centroids)):
                d_x = abs(i[0] - centroids[j][0])
                d_y = abs(i[1] - centroids[j][1])
                dist = (d_x**2+d_y**2)**1/2
                if dist !=0:
                    if dist < smallest:
                        smallest = dist
            distances.append(smallest)

        res = sorted(range(len(distances)),key=lambda sub:distances[sub],reverse=True)[:4]
        fin_corners = centroids[res]

        border = cv.Canny(img_gray,100,200)
        border_coords = np.where(border>0)

        for i in range(len(border_coords[0])):
            x = border_coords[0]
            y = border_coords[1]
            for i in range(4):
                corner = fin_corners[i]
                
                

        # plt.figure(1)
        # for i in fin_corners:
        #     plt.scatter(i[0],i[1])
        # plt.imshow(img)
        # plt.figure(2)
        # plt.imshow(dst)
        # plt.figure(3)
        # for i in centroids:
        #     plt.scatter(i[0],i[1])
        # plt.imshow(img)
        # plt.title('2')
        # plt.figure(4)
        # plt.imshow(border)
        # plt.show()