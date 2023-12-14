from scipy.spatial.distance import directed_hausdorff

def haus_dsit(edge0,edge1):
    dist = directed_hausdorff(edge0,edge1)
    return dist