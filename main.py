from split_edges import splitEdges
from hist_distance import histogram_intersect
from haus_dist import haus_dist
from matplotlib import pyplot as plt
import cv2 as cv
import os
import json
import numpy as np

class Piece:
    def __init__(self,name):
        self.edges = []
        self.concavity = []
        self.edge_paired = []
        self.name = name
        self.corner = False
        self.border = False
    
folder_dir = "C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\JigsawTrainingData\\using\\"
img_dir = "\\raster\\"
mask_dir = "\\mask\\"

fname = 'piece_id_to_mask.json'
# output_dir = "C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\outputs"

def main(pic_name):
    file = open(folder_dir + pic_name + '\\piece_id_to_mask.json')
    json_file = json.load(file)
    img_names = os.listdir(folder_dir + pic_name + '\\' + img_dir)
    img_names_copy = img_names.copy()

    # get edges
    pieces = []
    for i in range(len(img_names)):
        print(img_names[i])
        piece_n = Piece(img_names[i])
        img = cv.imread(folder_dir + pic_name + '\\' + img_dir+img_names[i])
        mask_name = json_file[img_names[i][:-4]]
        mask = cv.imread(folder_dir + pic_name + '\\' + mask_dir + mask_name + '.bmp')
        mask = cv.resize(mask,(img.shape[1],img.shape[0]))
        mask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
        concave, convex, straight = splitEdges.get_edges(mask)
        for i in range(4):
            if len(concave[i])>1:
                piece_n.edges.append(concave[i])
                piece_n.concavity.append(1)
            elif len(convex[i])>1:
                piece_n.edges.append(convex[i])
                piece_n.concavity.append(-1)
            else:
                piece_n.edges.append(straight[i])
                piece_n.concavity.append(2)
        pieces.append(piece_n)
        
    print("Edges Done")
    # for i in pieces:
    #     print((i.name,i.concavity))

    marked = np.zeros(len(img_names))
    for i in range(len(img_names)):
        intersections = []
        dist = []
        if marked[i]==0:
            img = cv.imread(folder_dir + pic_name + '\\' + img_dir+img_names[i])   
            intersections = []
            for n in range(len(img_names)):
                dst = cv.imread(folder_dir + pic_name + '\\' + img_dir+img_names_copy[n])
                if n!=i and marked[n]==0:
                    intersections.append(histogram_intersect(img,dst))
                else:
                    intersections.append(0)    
        intersections = np.array(intersections)
        intersections[np.where(intersections<max(intersections)*0.7)] = 0
        for p in range(len(intersections)):
            if p==i:
                dist.append(pieces[i])
            elif intersections[p]!=0:
                dist.append(pieces[p])
            else:
                dist.append(0)
        pic = haus_dist(dist)

    piece_dict = {}
    for i in pieces:
        piece_dict[i.name[:-4]] = i.edge_paired


    print(piece_dict)
    with open('C:\\Users\\evanj\\Desktop\\School\\Graduate\\Fall2023\\ComputerVision\\JigsawTermProject\\outputs\\'+ pic_name + ".json",'w') as f:
        json.dump(piece_dict,f)

# for pic_name in os.listdir(folder_dir):
#     print(pic_name)
#     main(pic_name)

main('test')