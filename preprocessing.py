#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:13:29 2022

@author: usn
"""
import cv2
import numpy as np

def detect_ver_hor_lines(kernel_len, flag):
    if flag == 1:
        v_or_h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    else:
        v_or_h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        
    return v_or_h_kernel
    
def detect_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def sort_contours(contour, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0    # handle if we need to sort in reverse
    if(method == "right-to-left" or method == "bottom-to-top"):
        reverse = True    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in contour]
    (contour, boundingBoxes) = zip(*sorted(zip(contour, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))    # return the list of sorted contours and bounding boxes
    return (contour, boundingBoxes)

def cell_detection(img_vh):
    # Detect contours for following box detection
    contours, hierarchy = detect_contours(img_vh)
    
    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    
    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    
    #Get mean of heights
    mean = np.mean(heights)
    
    #Create list box to store all boxes in  
    box = []# Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<1000 and h<500):
            # image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            box.append([x,y,w,h])
            
    return box,mean

def get_column_row(box, mean):
    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    # j=0
    #Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if(i==0):
            column.append(box[i])
            previous=box[i]    
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]            
                if(i==len(box)-1):
                    row.append(column)        
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])
                
    return row, column

def list_finalboxes(row):
    #calculating maximum number of cellscountcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol
    
    #Retrieving the center of each column
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
    center=np.array(center)
    center.sort()
    
    #Regarding the distance to the columns center, the boxes are arranged in respective order
    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)
        
    return finalboxes, countcol