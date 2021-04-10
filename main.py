import argparse
import os
import datetime

import skimage.io as io
import cv2
import numpy as np
import math  as m
import matplotlib as mplt
import copy


from skimage            import data
from skimage.filters    import threshold_otsu
from skimage.color      import rgb2gray
from skimage.feature    import canny
from skimage.transform  import hough_line,hough_line_peaks,rotate,resize
from skimage.morphology import binary_closing,binary_dilation,binary_erosion,binary_opening
from skimage.feature    import match_template #allowed
from skimage            import measure
from skimage.measure    import find_contours
from skimage.draw       import rectangle
from collections        import Counter

def binarize(image):
    image = rgb2gray(image)
    threshold = threshold_otsu(image)
    binary = image > threshold
    return binary


def deskew(image):
    original_image = image
    image = binarize(image)
    canny_image = canny(image)
    
    tested_angles = np.linspace(-np.pi/2, np.pi/2 , 90) #Test from -pi/2 to pi/2 with an increment dividing  range into 90 increments
    h, theta, d = hough_line(canny_image, theta=tested_angles) 
    nh,nt,nd =  hough_line_peaks(h, theta, d)
    opp_theta=  np.average(nt)
    opp_theta=  m.degrees(opp_theta)
    if (opp_theta == 90):
        opp_theta = -90
    rotated_image = rotate(original_image,opp_theta+90, resize=True)
    
    return rotated_image

def RemoveLines(img):
    image = binarize(img)
    f_h=np.ones((1, round(image.shape[1]/5)), dtype=int) #Horizontal kernel based on image width
    image_h = binary_closing(image,f_h) 
    image_h = binary_erosion(image_h,f_h) #Connect stafflines which have gaps
    contours = measure.find_contours(image_h, 0.1) 
    
    heights = [] 
    count=0
    height = 0
    for y in range(image.shape[0]): #in range(image height)
        if (image_h[y][1] == 1 and count == 0):
            height=0
        elif (image_h[y][1] == 0):
            height=height+y
            count=count+1
        elif (image_h[y][1] == 1 and count != 0): 
            heights.append(round(height/count))
            count=0
    
    y=0
    thickness=0
    while (image_h[y][1] == 1):
        y=y+1
    while (image_h[y][1] != 1):
        thickness=thickness+1
        y=y+1
    
    bounding_boxes = []
    for contour in contours:
        Xmin = int(round(np.min(contour[:,1])))
        Xmax = int(round(np.max(contour[:,1])))
        Ymin = int(round(np.min(contour[:,0])))
        Ymax = int(round(np.max(contour[:,0]))) 
        if ((Xmax-Xmin) > 0.8*image.shape[1]):
            bounding_boxes.append([Xmin, Xmax, Ymin-round(thickness/2)-2, Ymax+round(thickness/2)+2])

    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        for y in range(Ymin,Ymax,1):
            for x in range(Xmin,Xmax,1):
                if (Ymin-1 >= 0 and Ymax+1 <= image.shape[0]):
                    if (image[Ymin-1][x] == 1 and image[Ymax+1][x] == 1):
                        image[y,x] = 1
                    
    return image,heights

def get_ref_lengths(img):
    num_rows = img.shape[0]  # Image Height (number of rows)
    num_cols = img.shape[1]  # Image Width (number of columns)
    rle_image_white_runs = []  # Cumulative white run list
    rle_image_black_runs = []  # Cumulative black run list
    sum_all_consec_runs = []  # Cumulative consecutive black white runs

    for i in range(num_cols):
        col = img[:, i]
        rle_col = []
        rle_white_runs = []
        rle_black_runs = []
        run_val = 0  
        run_type = col[0]  
        for j in range(num_rows):
            if (col[j] == run_type):
                run_val += 1
            else:
                rle_col.append(run_val)
                if (run_type == 0):
                    rle_black_runs.append(run_val)
                else:
                    rle_white_runs.append(run_val)

             
                run_type = col[j]
                run_val = 1

        rle_col.append(run_val)
        if (run_type == 0):
            rle_black_runs.append(run_val)
        else:
            rle_white_runs.append(run_val)

        # Calculate sum of consecutive vertical runs
        sum_rle_col = [sum(rle_col[i: i + 2]) for i in range(len(rle_col))]

        # Add to column accumulation list
        rle_image_white_runs.extend(rle_white_runs)
        rle_image_black_runs.extend(rle_black_runs)
        sum_all_consec_runs.extend(sum_rle_col)

    white_runs =Counter(rle_image_white_runs)
    black_runs = Counter(rle_image_black_runs)
    black_white_sum = Counter(sum_all_consec_runs)

    line_spacing = white_runs.most_common(1)[0][0]
    line_height = black_runs.most_common(1)[0][0]
    width_spacing_sum = black_white_sum.most_common(1)[0][0]

   
    return line_height, line_spacing
    
    
## Hazem's code starts here
def segment_notes(testimg):
    img2 = testimg.copy()
    img2 = -1*img2
    y_dim = img2.shape[0]
    x_dim = img2.shape[1]
 

    st_el = np.ones((3,3))

    closed = binary_closing(testimg, st_el)

    contours = find_contours(closed, 0.8)
    bounding_boxes = []
    for contour in contours:
        Xmin = int(min(contour[:,1]))
        Xmax = int(max(contour[:,1]))
        Ymin = int(min(contour[:,0]))
        Ymax = int(max(contour[:,0]))
        width  = (Xmax - Xmin) 
        height = (Ymax - Ymin)
        aspect_ratio = abs(width/height)
        '''
            this part was supposed to fix multiple notes by detecting the aspect ratio and dividing based on it
            but it didn't work
        '''
#         if (aspect_ratio > 1):
# #             print("ar = ",aspect_ratio)
#             num_notes = math.floor(aspect_ratio/0.5)
#           #  print(num_notes)
#             patch = (Xmax+Xmin)/num_notes
#             patch = patch - Xmin
#             for k in range(0,num_notes):
#                 bounding_boxes.append([int(Xmin+k*patch)-1, int(Xmin+2*k*patch)-1, Ymin, Ymax])
#                # print(int(Xmin+k*patch)-1, int(Xmin+(k+1)*patch-1), Ymin, Ymax)
        if (aspect_ratio >= 0.1 and aspect_ratio <= 10):
            bounding_boxes.append([Xmin, Xmax, Ymin, Ymax])

    #When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
    i = 0
    newimages = []
    heights = []
    maxheights = []
    for box in bounding_boxes:
        newimg = np.ones(((Ymax-Ymin)+10,(Xmax-Xmin)+10))
        newimg[0:Ymax-Ymin,0:Xmax-Xmin]=testimg[Ymin:Ymax,Xmin:Xmax]
        newimages.append(newimg)
        heights.append(Ymax)
        maxheights.append(Ymin)
        [Xmin, Xmax, Ymin, Ymax] = box
        rr, cc = rectangle(start = (Ymin,Xmin), end = (Ymax,Xmax))
        img2[rr, cc] = 0 #set color white
    return newimages,heights,maxheights


def classify_notes(notes, heights,maxheights, SLH, SSH,filename):
    measuring_height = 0
    file1 = open(filename,"w+") 
    for i in range(len(notes)):
            templates = [
            "invertedclef.jpg",
            "clef.jpg",
            "dot.jpg",
            "Doubleflat.jpg",
            "doublesharp.jpg",
            "eight.jpg",
            "half.jpg",
            "quarter.jpg",
            "sharp.jpg",
            "Sixteen.jpg",
            "thirtysecond.jpg",
            "whole.jpg",
            "halfflip.jpg",
            "quarterflip.jpg"]

            class_names = [
                          "iclef",
                          "clef",
                          ".",
                          "&&",
                          "##",
                          "1/8",
                          "1/2",
                          "1/4",
                          "#",
                          "1/16",
                          "1/32",
                          "1/1",
                          "flippedh",
                          "flippedq"
                          ]
            ratings = []

            for temp in templates:
                str1 = "notes/" + temp
                t1 = io.imread(str1)
                resized = resize(t1,(int(notes[i].shape[0]-1),int(notes[i].shape[1]-1)))
                resized = rgb2gray(resized)
                ratio = match_template(notes[i], resized)
                val = sum(sum(ratio))
                ratings.append(val)

            note_index = ratings.index(max(ratings))
            note_name = class_names[note_index]
            if note_name == "clef":
                measuring_height = heights[i]
                file1.write("]\n[")

                '''
                    SLH = STAFFLINE HEIGHT = 2
                    SSH = STAFFSPACE HEIGHT = 20
                '''
            elif note_name == "flippedh":
                note_name = "1/2"
                if 3*(SLH+SSH) < measuring_height-maxheights[i]<4*(SLH+SSH)-3:
                    note_name =  "d" + note_name 
                if 4*(SLH+SSH)-3 < measuring_height-maxheights[i]<4*(SLH+SSH)+3:
                    note_name =  "e" + note_name 
                if 4*(SLH+SSH)+3 < measuring_height-maxheights[i]<5*(SLH+SSH)-3:
                    note_name =  "f" + note_name 
                if 5*(SLH+SSH)-3 < measuring_height-maxheights[i]<5*(SLH+SSH)+3:
                    note_name =  "g" + note_name 
                if 5*(SLH+SSH)+3 < measuring_height-maxheights[i]:
                    note_name =  "a" + note_name 
            elif note_name == "flippedq":
                note_name = "1/4"
                if 3*(SLH+SSH) < measuring_height-maxheights[i]<4*(SLH+SSH)-3:
                    note_name =  "d" + note_name 
                if 4*(SLH+SSH)-3 < measuring_height-maxheights[i]<4*(SLH+SSH)+3:
                    note_name =  "e" + note_name 
                if 4*(SLH+SSH)+3 < measuring_height-maxheights[i]<5*(SLH+SSH)-3:
                    note_name =  "f" + note_name 
                if 5*(SLH+SSH)-3 < measuring_height-maxheights[i]<5*(SLH+SSH)+3:
                    note_name =  "g" + note_name 
                if 5*(SLH+SSH)+3 < measuring_height-maxheights[i]:
                    note_name =  "a" + note_name 
            else:    
                if measuring_height-heights[i] < SLH:
                    note_name =  "e" + note_name
                if SLH <= measuring_height-heights[i] < SLH+SSH:
                    note_name =  "f" + note_name
                if SLH+SSH <= measuring_height-heights[i] < 2*SLH-5:
                    note_name =  "g" + note_name
                if 2*SLH-5 <= measuring_height-heights[i] < 3*SLH:
                    note_name =  "a" + note_name
            out1 = note_name+" "
            file1.write(out1)
    file1.write("]")
    file1.close()


parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")

args = parser.parse_args()
    
in_directory = args.inputfolder
out_directory = args.outputfolder


for file in os.listdir(in_directory):
    input_filename = os.fsdecode(os.path.join(in_directory, file)) #Input filename for reading
    print(input_filename)
    img = io.imread(input_filename)
    
    
    filename_without_ext, file_extension = os.path.splitext(file)
    print('file name without extension \n',filename_without_ext)
    
    
    output_filename = os.fsdecode(os.path.join(out_directory, filename_without_ext + '.txt'))
    print('output file name : \n',output_filename)

    
    img = binarize(img)
    img = deskew(img)
    lh, ls = get_ref_lengths(img)
    img,_ = RemoveLines(img)
    n, h, mh = segment_notes(img)  
    classify_notes(n, h, mh, lh, ls,output_filename)

     
