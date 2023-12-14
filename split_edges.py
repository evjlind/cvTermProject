import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon as poly

class splitEdges():

    def get_edges(img):
        # add padding to img
        img, new_img_w, new_img_h = splitEdges.add_padding(img)

        #pre-process
        #mask out
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        _, img_gray = cv.threshold(img_gray,13,255,cv.THRESH_BINARY)
        img_gray2 = cv.morphologyEx(img_gray, cv.MORPH_CLOSE,(3,3))
        img_gray2 = cv.morphologyEx(img_gray2,cv.MORPH_OPEN,(3,3))
        img_gray2 = cv.GaussianBlur(img_gray2,(3,3),0)
        
        #corner detection
        dst = cv.cornerHarris(img_gray2,2,3,0.02) # 0.04, 0.08

        dst[np.where(dst<0)] = 0
        dst *= 255 # or any coefficient
        dst = dst.astype(np.uint8)
        dst = cv.morphologyEx(dst,cv.MORPH_DILATE,(3,3))
        _, dst = cv.threshold(dst,0,255,cv.THRESH_OTSU+cv.THRESH_BINARY)
        _, _, _, centroids = cv.connectedComponentsWithStats(dst)
        centroids = centroids[1:]
        border = np.array(cv.Canny(img_gray2,100,200))
        border_coords = np.argwhere((border>0))
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
        fin_corners = splitEdges.compute_best_rectangle(centroids,0)
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
        temp = np.zeros_like(img_gray2)
        for i in range(len(fin_corners)-1):
            p1 = (int(fin_corners[i][0]),int(fin_corners[i][1]))
            p2 = (int(fin_corners[i+1][0]),int(fin_corners[i+1][1]))
            temp = cv.line(temp,(int(fin_corners[-1][0]),int(fin_corners[-1][1])),(int(fin_corners[0][0]),int(fin_corners[0][1])),1,1)
            
            temp = cv.line(temp,p1,p2,1,1)

        temp = cv.line(temp,(int(fin_corners[-1][0]),int(fin_corners[-1][1])),(int(fin_corners[0][0]),int(fin_corners[0][1])),1,1)

        #for every border pixel, associate it with its edge
        edges = splitEdges.div_edges(border,fin_corners)
        edge_images = [np.zeros_like(img_gray),np.zeros_like(img_gray),np.zeros_like(img_gray),np.zeros_like(img_gray)]
        for j in range(len(edge_images)):
            coords = edges[j]
            for i in coords:
                edge_images[j][i[0][1]][i[0][0]] = 1
        # find concavity of each side
        concave,convex,straight = splitEdges.determine_concavity(edge_images,temp)
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
            area0 = edges[i]
            area0 = cv.line(edges[i],(coords[1][0],coords[0][0]),(coords[1][-1],coords[0][-1]),1,1)
            edge_fill = area0.copy()
            h,w = area0.shape[:2]
            mask = np.zeros((h+2,w+2),np.uint8)
            cv.floodFill(edge_fill,mask,(0,0),(255,255))
            edge_fill = cv.bitwise_not(edge_fill)
            intersection = cv.bitwise_and(edge_fill,edge_fill,mask=rect_fill)
            intersection -= area0
            contours,_ = cv.findContours(intersection, 1, 2)
            try:
                cnt = contours[0]
                areas.append(cv.contourArea(cnt))
            except:
                areas.append(0)
        for i in range(len(areas)):
            if areas[i]>=max(areas)*0.5:
                concave.append(edges[i])
                convex.append([0])
                straight.append([0])
            elif areas[i] <= max(areas)*0.2:
                concave.append([0])
                convex.append([0])
                straight.append(edges[i])
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
        _, _, channels = img.shape
        new_img_w = 150
        new_img_h = 150
        result = np.full((new_img_h,new_img_w,channels),(0,0,0),dtype=np.uint8)
        x_center = (new_img_w - img.shape[1]) // 2
        y_center = (new_img_h - img.shape[0]) // 2

        result[y_center:y_center+img.shape[0],
            x_center:x_center+img.shape[1]] = img
        return result, new_img_w, new_img_h

    def div_edges(border, corners):
        n_corners = np.zeros_like(corners)
        for i in range(len(corners)):
            n_corners[i][0],n_corners[i][1] = int(corners[i][0]),int(corners[i][1])
        temp = np.zeros_like(border)
        temp = cv.line(temp,(int(n_corners[0][0])-10,int(n_corners[0][1])-10),(int(n_corners[2][0])+10,int(n_corners[2][1])+10),255,2)
        temp = cv.line(temp,(int(n_corners[1][0])+10,int(n_corners[1][1])-10),(int(n_corners[3][0])-10,int(n_corners[3][1])+10),255,2)
        divider = cv.bitwise_not(temp)
        final = cv.bitwise_and(border,border,mask=divider)
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
                    if (1.553 - toler) < theta1 < (1.588 + toler):
                        for m in range(j,len(points)):
                            side_vect2 = (points[j][0]-points[m][0],points[j][1]-points[m][1])
                            side_vect3 = (points[m][0]-points[i][0],points[m][1]-points[i][1])
                            theta2 = np.arccos(np.dot(side_vect1,side_vect2)/(splitEdges.vector_mag(side_vect2)*splitEdges.vector_mag(side_vect1)))
                            if (1.553 - toler) < theta2 < (1.588 + toler):
                                theta3 = np.arccos(np.dot(side_vect2,side_vect3)/(splitEdges.vector_mag(side_vect3)*splitEdges.vector_mag(side_vect2)))
                                if (1.553 - toler) < theta3 < (1.588 + toler):
                                        theta4 = np.arccos(np.dot(side_vect3,base_vect)/(splitEdges.vector_mag(side_vect3)*splitEdges.vector_mag(base_vect)))
                                        if (1.553 - toler) < theta4 < (1.588 + toler):
                                            x = [points[i][0],points[k][0],points[j][0],points[m][0]]
                                            y = [points[i][1],points[k][1],points[j][1],points[m][1]]
                                            rect = poly(zip(x,y))
                                            if rect.area>max_area:
                                                best = [points[i],points[j],points[k],points[m]]
                                                max_area = rect.area
        if max_area == 0:                                   
            best = splitEdges.compute_best_rectangle(points,toler+0.001)
        return best
                        
    def vector_mag(vector):
        return np.sqrt(vector[0]**2+vector[1]**2)
    