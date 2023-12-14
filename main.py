from split_edges import splitEdges
from hist_distance import histogram_intersect
from haus_dist import haus_dsit
from matplotlib import pyplot as plt
import cv2 as cv
import os
import json
import numpy as np

img_dir = "C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\JigsawTrainingData\\testpieces\\test1\\size-100\\raster\\"
mask_dir = "C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\JigsawTrainingData\\testpieces\\test1\\size-100\\mask\\"
json_dir = "C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\JigsawTrainingData\\testpieces\\test1\\size-100\\"

fname = 'piece_id_to_mask.json'
file = open(json_dir + fname)
json_file = json.load(file)
toler = 0.7
img_names = os.listdir(img_dir)
img_names_copy = img_names.copy()
adjacency_matrix = np.zeros((len(img_names),len(img_names)))
edges = [[],[],[]]
# get edges
for i in range(len(img_names)):
    img = cv.imread(img_dir+img_names[i])
    mask_name = json_file[img_names[i][:-4]]
    mask = cv.imread(mask_dir + mask_name + '.bmp')
    mask = cv.resize(mask,(img.shape[1],img.shape[0]))
    mask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
    img2 = cv.bitwise_and(img,img,mask=mask)
    img2,_,_ = splitEdges.add_padding(img2)
    concave, convex, straight = splitEdges.get_edges(cv.imread(img_dir+img_names[i]))
    edges[0].append(concave)
    edges[1].append(convex)
    edges[2].append(straight)
plt.imshow(edges[0][0][0])
plt.show()
# for i in range(len(img_names)):
#     img = cv.imread(img_dir+img_names[i])   
#     intersections = []
#     for n in range(len(img_names)):
#         dst = cv.imread(img_dir+img_names_copy[n])
#         if n!=i and sum(adjacency_matrix[n])==0:
#             intersections.append(histogram_intersect(img2,dst))
#         else:
#             intersections.append(0)
#     edges = [[],[],[]]  #edge[concavity][piece][side]
#     print("Intersections Done")
#     for j in range(len(intersections)):
#         if j>=(toler*max(intersections)):
#             # print(img_names[j])
#             pass

border_pieces = np.zeros_like(img_names)
distances = [[],[],[]]
# for i in range(len(edges[2])):
#     for j in range(len(edges[2][i])):
#         # n = edges[2][i][j]
#         for k in range(len(edges[2][i][j])):
#             print(k)
#             plt.figure(1)
#             plt.show()

# print(border_pieces)
# print(len(border_pieces))

# # for i in edges[0]:
# #     for j in edges[1]:
# #         distances[0].append(haus_dsit(i,j))
# #         break
# # print(distances[0])


    #edges order: left, top, right, bottom

    # img_names_copy.remove(img_names_copy[i])
    

# for fname in os.listdir(img_dir):
#     print(fname)
#     img = cv.imread(img_dir + '/' + fname)
#     splitEdges.get_edges(img)