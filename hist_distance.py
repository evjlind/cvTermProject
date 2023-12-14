import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def histogram_intersect(piece0,piece1):
    src = piece0
    dst = piece1
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    hsv_base = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    hsv_comp = cv.cvtColor(dst,cv.COLOR_BGR2HSV)
    hist_base = np.delete(cv.calcHist([hsv_base],channels,None,histSize,ranges,accumulate=False),[[0,0],[1,0]])
    cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    hist_comp = np.delete(cv.calcHist([hsv_comp],channels,None,histSize,ranges,accumulate=False),[[0,0],[1,0]])
    cv.normalize(hist_comp, hist_comp, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    intersection = cv.compareHist(hist_base,hist_comp,cv.HISTCMP_INTERSECT)
    return intersection