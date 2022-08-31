#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:13:28 2022

@author: usn
"""
import cv2
import numpy as np

from preprocessing import detect_ver_hor_lines

def otsu_thresh(img):
    thresh,img_out = cv2.threshold(img,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    return thresh,img_out

def erode(img_bin, ver_hor_kernel, iterations=3):
    erode_img = cv2.erode(img_bin, ver_hor_kernel, iterations=iterations)
    return erode_img

def dilate(img, ver_hor_kernel, iterations=3):
    out_lines = cv2.dilate(img, ver_hor_kernel, iterations=iterations)
    return out_lines

def filter_in_image(img):
    #thresholding the image to a binary image
    thresh,img_bin = otsu_thresh(img)
    
    #inverting the image 
    img_bin = 255-img_bin
        
    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100
    
    # Defining a vertical kernel to detect all vertical lines of image
    # set flag to 0 for vertical line detection
    ver_kernel = detect_ver_hor_lines(kernel_len, 0)
    
    # Defining a horizontal kernel to detect all horizontal lines of image
    # set flag to 1 for vertical line detection
    hor_kernel = detect_ver_hor_lines(kernel_len, 1)
    
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    #Use vertical kernel to detect and save the vertical lines
    image_1 = erode(img_bin, ver_kernel)
    vertical_lines = dilate(image_1, ver_kernel)
        
    #Use horizontal kernel to detect and save the horizontal lines 
    image_2 = erode(img_bin, hor_kernel)
    horizontal_lines = dilate(image_2, hor_kernel)
    
    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    
    #Eroding and thesholding the image
    img_vh = erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = otsu_thresh(img_vh)
    
    return img_vh
