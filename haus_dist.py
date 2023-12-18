from scipy.spatial.distance import directed_hausdorff
import numpy as np

def haus_dist(pieces):
    concave, convex = [],[]
    for i in pieces:
        straight = 0
        if i!=0:
            piece_cons = i.concavity
            piece_edges = [[],[],[],[]]
            for n in range(len(i.edges)):
                piece_edges[n] = np.where(i.edges[n]!=0)
            for n in range(len(piece_cons)):
                if len(piece_edges[n])>1:
                    piece_x = piece_edges[n][0] - min(piece_edges[n][0])
                    piece_y = piece_edges[n][1]- min(piece_edges[n][0])
                    edge = np.column_stack((piece_x,piece_y))
                    if piece_cons[n]==1:
                        concave.append((edge,i))
                    elif piece_cons[n]==-1:
                        convex.append((edge,i))
                    else:
                        straight += 1
            if straight == 1:
                i.border = True
            elif straight == 2:
                i.corner = True
    
    pairs = []
    for i in concave:
        edge1 = i[0]
        piece1 = i[1]
        for p in convex:
            edge2 = p[0]
            piece2 = p[1]
            if piece1 != piece2:
                dist = directed_hausdorff(edge1,edge2)[0]
                pair = (piece1,piece2,dist)
                pairs.append(pair)      
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if i!=j and pairs[i]!=[0,0,0] and pairs[j]!=[0,0,0]:
                if ((pairs[i][0]==pairs[j][0] and pairs[i][1]==pairs[j][1]) or (pairs[i][1]==pairs[j][0] and pairs[i][0]==pairs[j][0])):
                    if pairs[i][2]<pairs[j][2]:
                        pairs[j] = [0,0,0]
                    else:
                        pairs[i] = [0,0,0]
    sorted_pairs = sorted(pairs,key=lambda x:x[2]) 
    for i in sorted_pairs:
        if i!=[0,0,0]:
            if i[0].corner==True:
                if len(i[0].edge_paired)<2 and len(i[1].edge_paired)<2:
                    i[0].edge_paired.append(i[1].name[:-4])
                    i[1].edge_paired.append(i[0].name[:-4])
            elif i[0].border == True:
                if len(i[0].edge_paired)<3 and len(i[1].edge_paired)<3:
                    i[0].edge_paired.append(i[1].name[:-4])
                    i[1].edge_paired.append(i[0].name[:-4])
            else:
                if len(i[0].edge_paired)<4 and len(i[1].edge_paired)<4:
                    i[0].edge_paired.append(i[1].name[:-4])
                    i[1].edge_paired.append(i[0].name[:-4])
    return pieces