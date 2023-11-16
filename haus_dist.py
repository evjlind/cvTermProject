from scipy.spatial.distance import directed_hausdorff as haus_dist
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob, os
#harris corners, extract sides, overlay and then hausdorrf it?
# kernel = np.ones((3,3),np.float32)/9
# # for file in glob.glob("JigsawTrainingData\motorbike_0175\size-100\mask\*"):
# for file in glob.glob("JigsawTrainingData\motorbike_0175\*"):
#     if file.endswith(".jpg"):
#         print(file)
#         img = cv.imread(file)
#         cpy = np.copy(img)
#         img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#         img = cv.filter2D(img,-1,kernel)
#         dst = cv.cornerHarris(img,2,3,0.04)
#         threshold = .1*dst.max()
#         corner_img = np.copy(img)
#         for j in range(0, dst.shape[0]):
#             for i in range(0,dst.shape[1]):
#                 if dst[j,i] > threshold:
#                     cv.circle(corner_img, (i,j),1,(0,255,0),1)
#         plt.figure(1)
#         plt.imshow(corner_img)
#         plt.figure(2)
#         plt.imshow(cpy)
#         plt.show()
#dist = haus_dist(u,v)

img = cv.imread("JigsawTrainingData\MjU-padding.bmp")

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

# preprocessing
kernel_size = 3
kern = np.ones((3,3))/9

img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur_gray = cv.GaussianBlur(img_gray,(kernel_size,kernel_size),0)
corner_img = cv.cornerHarris(blur_gray,3,9,0.02)
corner_img = cv.dilate(corner_img,None,iterations=1)
t, corner_img = cv.threshold(corner_img,0.2,255,cv.THRESH_TOZERO)
corner_img = cv.erode(corner_img,kern)
corner_img = np.uint8(corner_img)

ret, labels, stats, centroids = cv.connectedComponentsWithStats(corner_img)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
corners = cv.cornerSubPix(img_gray,np.float32(centroids),(3,3),(-1,-1),criteria)

res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv.imwrite('subpixel5.png',img)

plt.figure(1)
plt.imshow(corner_img)
plt.figure(2)
plt.imshow(blur_gray)

plt.show()
