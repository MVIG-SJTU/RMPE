# -*- coding: utf-8 -*-
"""
Created on Wed Oct 5  2016

@author: Fang-Haoshu
"""

import os
import numpy as np
import math

def cropBox(image, xmin, ymin, xmax, ymax):
    cropped_image = image[ymin:ymax,xmin:xmax,:]
    
    width = xmax - xmin
    height = ymax - ymin
    length = max(width,height)
    pad_top = (length - height)/2
    pad_bottom = (length - height + 1)/2
    pad_left = (length - width)/2
    pad_right = (length - width + 1)/2
    padded_image = np.ndarray((length,length,3),np.float32)
    padded_image[:,:,0] = np.pad(cropped_image[:,:,0],((int(pad_top),int(pad_bottom)),(int(pad_left),int(pad_right))),'constant', constant_values=(0, 0))
    padded_image[:,:,1] = np.pad(cropped_image[:,:,1],((int(pad_top),int(pad_bottom)),(int(pad_left),int(pad_right))),'constant', constant_values=(0, 0))
    padded_image[:,:,2] = np.pad(cropped_image[:,:,2],((int(pad_top),int(pad_bottom)),(int(pad_left),int(pad_right))),'constant', constant_values=(0, 0))

    return padded_image


def transformBoxInvert(pt, xmin, ymin, xmax, ymax, res):
    center_x = (xmax-xmin)/2
    center_y = (ymax-ymin)/2

    length = max(ymax-ymin, xmax-xmin)

    _pt = [(x*length)/res for x in pt]

    _pt[0] = _pt[0]-math.floor(max(0,(length/2 - center_x)))
    _pt[1] = _pt[1]-math.floor(max(0,(length/2 - center_y)))

    new_point = [int(_pt[0]+xmin),int(_pt[1]+ymin)]
    return new_point
