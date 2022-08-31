#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:02:02 2022

@author: usn
"""

import cv2
import numpy as np
import pandas as pd
import pytesseract
import json

from image_filters import dilate, erode, filter_in_image
from preprocessing import cell_detection, get_column_row, list_finalboxes

image_path = r'images/input_image.png'

class main_ocr:
    def __init__(self):
        self.file = image_path
                   
    def tesseract_detect_string(self):
        #read your file
        img = cv2.imread(self.file, 0)
        
        # Applying multiple filters in image
        img_vh = filter_in_image(img)
        
        bitxor = cv2.bitwise_xor(img,img_vh)
        bitnot = cv2.bitwise_not(bitxor)
        
        box, mean = cell_detection(img_vh)        
        row, column = get_column_row(box, mean)
            
        finalboxes, countcol = list_finalboxes(row)
        
        #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
        outer=[]
        for i in range(len(finalboxes)):
            for j in range(len(finalboxes[i])):
                inner=''
                if(len(finalboxes[i][j])==0):
                    outer.append(' ')       
                else:
                    for k in range(len(finalboxes[i][j])):
                        y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                        finalimg = bitnot[x:x+h, y:y+w]
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                        border = cv2.copyMakeBorder(finalimg,2,2,2,2,   cv2.BORDER_CONSTANT,value=[255,255])
                        resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        
                        dilation = dilate(resizing, kernel,iterations=1)
                        erosion = erode(dilation, kernel,iterations=1)    
                        
                        out = pytesseract.image_to_string(erosion)
                        if(len(out)==0):
                            out = pytesseract.image_to_string(erosion, config='--psm 3')
                        inner = inner +" "+ out
                        outer.append(inner)
        return row, countcol, outer
                        
    def to_dataframe(self):
        row, countcol, outer = self.tesseract_detect_string()        
        #Creating a dataframe of the generated OCR list
        arr = np.array(outer)
        len_row = len(row)
        dataframe = pd.DataFrame(arr.reshape(len_row, countcol))
        # print(dataframe)
        
        data_dict = {}
        for i in range(1,len(dataframe)):
            
            dataframe[1][i] = dataframe[1][i].strip('\f')
            dataframe[1][i] = dataframe[1][i].strip('\n')    
            dataframe[1][i] = dataframe[1][i].replace('\n', ' ')
            
            dataframe[2][i] = dataframe[2][i].strip('\f')
            dataframe[2][i] = dataframe[2][i].strip('\n')
            dataframe[2][i] = dataframe[2][i].replace('\n', ' ')
            
            data_dict[dataframe[1][i]] = dataframe[2][i]
                        
        return data_dict
   

if __name__ == "__main__":
    main_obj = main_ocr()
    data_dict = main_obj.to_dataframe()
    
        
    json_obj = json.dumps(data_dict, indent=4)
    # print(json_obj)
    
    with open("json_final_data.json", "w") as json_file:
        json_file.write(json_obj)
        json_file.close()
    






























