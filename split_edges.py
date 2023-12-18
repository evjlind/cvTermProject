import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon as poly

class splitEdges():

    def get_edges(img):
        # add padding to img
        img, new_img_w, new_img_h = splitEdges.add_padding(img)
        #pre-process
        # img_gray2 = cv.GaussianBlur(img,(3,3),0)
        sharp_kern = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img = cv.filter2D(img,-1,sharp_kern)
        #corner detection
        dst = cv.cornerHarris(img,5,5,0.02) # 0.04, 0.08
        dst[dst<dst.max()*0.3] = 0
        dst *= 255 # or any coefficient
        dst = dst.astype(np.uint8)
        dst = cv.GaussianBlur(dst,(3,3),0)
        dst = cv.morphologyEx(dst,cv.MORPH_ERODE,(5,5))
        #dst = cv.morphologyEx(dst,cv.MORPH_DILATE,(6,6))
        #_, dst = cv.threshold(dst,0,255,cv.THRESH_OTSU+cv.THRESH_BINARY)
        _, _, _, centroids = cv.connectedComponentsWithStats(dst)
        centroids = centroids[1:]
        #border = np.array(cv.Canny(img_gray2,100,200))
        img = cv.GaussianBlur(img,(3,3),0)
        border = np.array(cv.Canny(img,0,200))

        # plt.imshow(dst)
        # plt.show()

        #order corners in polar order to properly draw lines
        polar_corners = []
        for i in centroids:
            polar_corners.append(splitEdges.cart2pol(np.floor(i[0])-new_img_w/2,np.floor(i[1])-new_img_h/2))
        polar_corners = sorted(polar_corners,key=lambda x:x[1])
        n=0
        for i in polar_corners:
            x,y = splitEdges.pol2cart(i[0],i[1])
            x,y = x+new_img_w/2,y+new_img_h/2
            centroids[n] = (x,y)
            n+=1
        try:
            fin_corners = splitEdges.compute_best_rectangle(centroids,0)
        except:
            print("piece fail")
            concave,convex,straight = [[0],[0],[0],[0]],[[0],[0],[0],[0]],[[0],[0],[0],[0]]
            return concave,convex,straight
        #order corners in polar order to properly draw lines
        polar_corners = []
        for i in fin_corners:
            polar_corners.append(splitEdges.cart2pol(np.floor(i[0])+0.5-new_img_w/2,np.floor(i[1])+0.5-new_img_h/2))
        polar_corners = sorted(polar_corners,key=lambda x:x[1])
        n=0
        for i in polar_corners:
            x,y = splitEdges.pol2cart(i[0],i[1])
            x,y = x+new_img_w/2,y+new_img_h/2
            fin_corners[n] = (x,y)
            n+=1
        #draw lines between corners
        temp = np.zeros_like(img)
        for i in range(len(fin_corners)-1):
            p1 = (int(fin_corners[i][0]),int(fin_corners[i][1]))
            p2 = (int(fin_corners[i+1][0]),int(fin_corners[i+1][1]))
            temp = cv.line(temp,(int(fin_corners[-1][0]),int(fin_corners[-1][1])),(int(fin_corners[0][0]),int(fin_corners[0][1])),1,1)

            temp = cv.line(temp,p1,p2,1,1)

        temp = cv.line(temp,(int(fin_corners[-1][0]),int(fin_corners[-1][1])),(int(fin_corners[0][0]),int(fin_corners[0][1])),1,1)

        # plt.imshow(img)
        # for i in fin_corners:
        #     plt.scatter(i[0],i[1])
        # plt.show()

        #for every border pixel, associate it with its edge
        edges = splitEdges.div_edges(border,fin_corners)
        edge_images = [np.zeros_like(img),np.zeros_like(img),np.zeros_like(img),np.zeros_like(img)]
        try:
            for j in range(4):
                coords = edges[j]
                for i in coords:
                    edge_images[j][i[0][1]][i[0][0]] = 1
            concave,convex,straight = splitEdges.determine_concavity(edge_images,temp)
        except:
            print('piece fail')
            concave,convex,straight = [[0],[0],[0],[0]],[[0],[0],[0],[0]],[[0],[0],[0],[0]]
        # find concavity of each side
        return concave, convex, straight

    def determine_concavity(edges,rectangle):
        concave,convex,straight = [],[],[]
        rect_fill = rectangle.copy()
        h,w = rectangle.shape[:2]
        mask = np.zeros((h+2,w+2),np.uint8)
        cv.floodFill(rect_fill,mask,(0,0),(255,255))
        rect_fill = cv.bitwise_not(rect_fill)
        areas = []
        for i in range(len(edges)):
            coords = np.where(edges[i]==1)
            if (max(coords[0])-min(coords[0]) > max(coords[1])-min(coords[1])):
                c1 = (coords[1][np.where(coords[0] == max(coords[0]))][0],max(coords[0]))
                c2 = (coords[1][np.where(coords[0] == min(coords[0]))][0],min(coords[0]))
            else:
                c1 = (max(coords[1]),coords[0][np.where(coords[1] == max(coords[1]))][0])
                c2 = (min(coords[1]),coords[0][np.where(coords[1] == min(coords[1]))][0])
            area0 = edges[i].copy()
            area0 = cv.line(area0,c1,c2,1,1)
            edge_fill = area0.copy()
            h,w = area0.shape[:2]
            mask = np.zeros((h+2,w+2),np.uint8)
            cv.floodFill(edge_fill,mask,(0,0),(255,255))
            edge_fill = cv.bitwise_not(edge_fill)
            intersection = cv.bitwise_and(edge_fill,edge_fill,mask=rect_fill)
            plt.imshow(intersection)
            plt.imshow(area0,alpha=0.3)
            plt.show()
            try:
                n_area = np.sum(intersection)
                o_area = np.sum(edge_fill)
                areas.append(n_area/o_area)
            except:
                areas.append(0)
        for i in range(len(areas)):
            if areas[i]==0:
                concave.append([0])
                convex.append([0])
                straight.append(edges[i])
            elif areas[i]>=0.6:
                concave.append(edges[i])
                convex.append([0])
                straight.append([0])
            else:
                concave.append([0])
                convex.append(edges[i])
                straight.append([0])
        return concave,convex,straight

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)

    def add_padding(img):
        new_img_w = 150
        new_img_h = 150
        result = np.full((new_img_h,new_img_w),0,dtype=np.uint8)
        x_center = (new_img_w - img.shape[1]) // 2
        y_center = (new_img_h - img.shape[0]) // 2

        result[y_center:y_center+img.shape[0],
            x_center:x_center+img.shape[1]] = img
        return result, new_img_w, new_img_h

    def div_edges(border, corners):
        border = cv.GaussianBlur(border,(1,1),0)
        _,border = cv.threshold(border,0,255,cv.THRESH_BINARY)
        n_corners = np.zeros_like(corners)
        for i in range(len(corners)):
            n_corners[i][0],n_corners[i][1] = int(corners[i][0]),int(corners[i][1])
        temp = np.zeros_like(border)
        temp = cv.line(temp,(int(n_corners[0][0])-10,int(n_corners[0][1])-10),(int(n_corners[2][0])+10,int(n_corners[2][1])+10),255,2)
        temp = cv.line(temp,(int(n_corners[1][0])+10,int(n_corners[1][1])-10),(int(n_corners[3][0])-10,int(n_corners[3][1])+10),255,2)
        divider = cv.bitwise_not(temp)
        final = cv.bitwise_and(border,border,mask=divider)
        # plt.imshow(final)
        # plt.show()
        n_edges, _ = cv.findContours(final,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        return n_edges

    def compute_best_rectangle(points,toler):
        max_area = 0
        for i in range(len(points)):
            for k in range(i,len(points)):
                base_vect = (points[i][0]-points[k][0],points[i][1]-points[k][1])
                for j in range(k,len(points)):
                    side_vect1 = (points[k][0]-points[j][0],points[k][1]-points[j][1])
                    theta1 = np.arccos(np.dot(base_vect,side_vect1)/(splitEdges.vector_mag(base_vect)*splitEdges.vector_mag(side_vect1)))
                    if (1.56 - toler) < theta1 < (1.58 + toler):
                        for m in range(j,len(points)):
                            side_vect2 = (points[j][0]-points[m][0],points[j][1]-points[m][1])
                            side_vect3 = (points[m][0]-points[i][0],points[m][1]-points[i][1])
                            theta2 = np.arccos(np.dot(side_vect1,side_vect2)/(splitEdges.vector_mag(side_vect2)*splitEdges.vector_mag(side_vect1)))
                            if (1.56 - toler) < theta2 < (1.58 + toler):
                                theta3 = np.arccos(np.dot(side_vect2,side_vect3)/(splitEdges.vector_mag(side_vect3)*splitEdges.vector_mag(side_vect2)))
                                if (1.56 - toler) < theta3 < (1.58 + toler):
                                        theta4 = np.arccos(np.dot(side_vect3,base_vect)/(splitEdges.vector_mag(side_vect3)*splitEdges.vector_mag(base_vect)))
                                        if (1.56 - toler) < theta4 < (1.58 + toler):
                                            x = [points[i][0],points[k][0],points[j][0],points[m][0]]
                                            y = [points[i][1],points[k][1],points[j][1],points[m][1]]
                                            rect = poly(zip(x,y))
                                            if rect.area>max_area:
                                                best = []
                                                for i in range(4):
                                                    best.append(np.array((x[i],y[i]),dtype=np.uint8))
                                                max_area = rect.area
        if max_area == 0:
            n_toler = toler + 0.01
            best= splitEdges.compute_best_rectangle(points,n_toler)
        return best

    def vector_mag(vector):
        return np.sqrt(vector[0]**2+vector[1]**2)


# 46, 55, 74, 78, 91