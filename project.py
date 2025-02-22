
# Import packages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd

# Folder for the dataset
folder = "codici-lineari-dati/"
file_name = "table.xlsx"

# Dictionary as database for the Principal Excel table
Principal_table = {
    "Images" : [],
    "Overall barcode grade" : [],
    "X-dimension": [],
    "Height" : [],
    "Left-up" : [], "Left-down" : [],"Right-up" : [],"Right-down" : [],
    "Center" : [],
    "Orientation" : []
}

# Creation of the excel file
with pd.ExcelWriter(file_name) as writer:
    # Scrivi il DataFrame vuoto nel file Excel
    pd.DataFrame().to_excel(writer, index=False)

# Get the list of files in the directory
img_list = [ x for x in os.listdir(folder) if ".BMP" in x]
print("List of images:\n {}" .format(img_list))

# morphological operations

# Opening - find patterns on the foreground and eliminate shapes not matching the pattern by placing them in the background.
def opening(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size))
    return cv2.dilate(cv2.erode(img, kernel),kernel)
    
# Closing - find patterns on the background and highlight shapes not matching the pattern by placing them in the foreground.
def closing(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size))
    return cv2.erode(cv2.dilate(img, kernel),kernel)

# Initialization for sub-plot of all the found barcode
_ , axs = plt.subplots(7,8)
print("Code running... Images are {}". format(len(img_list)))
# indices for visualize the sub-plots
ind = 0
j = 0

for element in img_list:
        
    # Image reading
    img = cv2.imread(folder + element, cv2.IMREAD_GRAYSCALE)

    # Shape of the image
    height_img, width_img = img.shape[:2]

    # Zero matrix as base for the new gradient image
    img_gradient = np.zeros(img.shape)

    # Sobel gradient computation - A smooth derivative that avoids blur but is robust with respect to noise. The Sobel operation also improves isotropy
    I_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) # Derivative along x
    I_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) # Derivative along y

    img_gradient = np.sqrt(I_x**2 + I_y**2)

    # Gaussian filter - We use it to blur the image and obtain thicker edges 
    sigma = 3
    kernel_size = int(np.ceil((3*sigma))*2+1) # the kernel is typically chosen of dimension 2k+1 x 2k+1 ; where k = 3*sigma

    gk = cv2.getGaussianKernel(kernel_size, sigma) # 1D Gaussian filter kernel
    gk_2D = gk.dot(gk.transpose()) # 2D Gaussian filter kernel

    # Computing the correlation between the image and the Filter Kernel - The Gaussian kernel is symmetric, so the correlation is equal to the convolution
    img_gaussian = cv2.filter2D(img_gradient, -1, gk_2D)

    # Thresholding the filtered gradient image with Otsu
    th,img_bin = cv2.threshold(img_gaussian.astype("uint8"),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    # cv2.THRESH_BINARY - The image will be binarized between 255 and 0
                                                                                                        # cv2.THRESH_OTSU - The threshold is automatically computed with Otsu's algorithm
    
    # Opening with a line pattern to remove unwanted elements

    # One manipulation is performed in the vertical direction, while the other is performed in the horizontal direction, resulting in two different images

    vertical = opening(img_bin, (2,60)) # Opening with vertical lines

    horizontal = opening(img_bin, (60,2)) # Opening with horizontal lines

    # Closing in order to obtain a uniform connected region for the barcode
    vertical = closing(vertical, (40,40))
    horizontal = closing(horizontal, (40,40))

    # Looking for connected region that has the highest area

    # The function computes connected components in the binary image along with statistics
    _ , _, stats_v, _ = cv2.connectedComponentsWithStats(vertical, connectivity = 8) # Connected regions in the vertical direction

    _ , _, stats_h , _ = cv2.connectedComponentsWithStats(horizontal, connectivity = 8) # Connected regions in the horizontal direction

    # Looking for the index of the element with the maximum area
    max_area_index_h = np.argmax(stats_h[1:,cv2.CC_STAT_AREA]) + 1 # starts from 1 because 0 is the background

    max_area_index_v = np.argmax(stats_v[1:,cv2.CC_STAT_AREA]) + 1 # start from 1 because 0 is the background

    if stats_h[max_area_index_h,cv2.CC_STAT_AREA] > stats_v[max_area_index_v,cv2.CC_STAT_AREA] :
        x, y, width, height, area = stats_h[max_area_index_h]
        is_vertical = False  # The barcode is considered not vertical when the maxinum area found with horizontal manipulations is grater than the one found with vertical manipulations
    else:
        x, y, width, height, area = stats_v[max_area_index_v]
        is_vertical = True # The barcode is considered vertical when the maxinum area found with horizontal manipulations is lower than the one found with vertical manipulations

    margin = 10 # Margin in ROI identification

    # Offset definition - Position of the ROI
    # I will use these quantities to crop the image in nexts manipulations
    offset_x = x - margin
    offset_y = y - margin
    ROI_offset = np.array([offset_y,offset_x]) # ROI left-upper corner

    ROI_width = width + 2*margin
    ROI_height = height + 2*margin
    ROI_shape =  np.array([ROI_height,ROI_width]) # ROI shape [height,width]

    # Image cropping the ROI
    barcode = img[ROI_offset[0]:ROI_offset[0] + ROI_shape[0], ROI_offset[1]: ROI_offset[1] + ROI_shape[1]]

    # Barcode binarization
    # Thresholding the Barcode ROI with Otsu
    th, barcode_bin = cv2.threshold(barcode.astype("uint8"),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # cv2.THRESH_BINARY - The image will be binarized between 255 and 0

    # Before the rotation, performing a closing (conceptually is an opening but given that the background is white, it is applied as a closing) to remove the unwanted elements (like numbers)
    if is_vertical: points_cloud = closing(barcode_bin,(2,60)) # if bars are vertical, the closing is vertical
    else: points_cloud = closing(barcode_bin,(60,2)) # Otherwise the closing is horizontal

    # Rotation 

    # Looking for points coordinates that compose the barcode
    x,y = np.nonzero(255 - points_cloud) # array of coordinates
    points = np.transpose(np.array([x,y])) # Formatting points to be interpreted correctly by minAreaRect()

    # Finding the oriented rectangle with minimum area that contains the barcode
    _ , size , theta = cv2.minAreaRect(points)
    
    # shape of the binarized barcode image
    height, width = barcode_bin.shape[:2]

    # Taking the correct angle respect to the horizontal
    if is_vertical: 
        if np.abs(theta) < 45 : theta = - theta
        else: theta =  90 - theta
    else: 
        if np.abs(theta) > 45 : theta = - theta
        else: theta = -(theta +90)

    # Computation of the rotation matrix respect to the center of the image
    rotation_matrix = cv2.getRotationMatrix2D([height/2,width/2],theta, 1.0)

    # Computation of the rotation matrix for the original image 
    rotation_matrix_img = cv2.getRotationMatrix2D([height_img/2,width_img/2], theta, 1.0)

    if np.abs(theta) > 45:
        # If theta is grater than 45° (horizontal bars), I need to reshape the image to avoid loosing information
        rotation_matrix[0, 2] += (height - width) / 2
        rotation_matrix[1, 2] += -(width - height) / 2
        # Rotation of the ROI
        barcode_oriented_bin = cv2.warpAffine(barcode_bin, rotation_matrix, (height,width), borderValue=(255,255,255)) # BorderValue is the color value of the padding - in this case is setted to white
        
        # If theta is grater than 45° (horizontal bars), I need to reshape also the original image to avoid loosing information
        rotation_matrix_img[0, 2] += (height_img - width_img) / 2
        rotation_matrix_img[1, 2] += -(width_img - height_img) / 2
        # Rotation of the image
        img_oriented = cv2.warpAffine(img, rotation_matrix_img, (height_img, width_img) )
        # Offset of the ROI inside the image - I will use these quantities to crop the image in nexts manipulations
        offset_y = ROI_offset[1] # y becomes the old x
        offset_x = height_img - ROI_offset[0] - ROI_shape[0] # x becomes the old y but starting from the bottom, so I need to subtract from the Image height the ROI height and the previous y.

    else:
        # Rotation of the ROI
        barcode_oriented_bin = cv2.warpAffine(barcode_bin, rotation_matrix, (width,height), borderValue=(255,255,255))
        # Rotation of the image
        img_oriented = cv2.warpAffine(img, rotation_matrix_img, (width_img,height_img))


    # Opening with a line pattern to remove unwanted elements
    # numbers elimination 

    barcode_opening = closing(barcode_oriented_bin, (2,60))
    height, width = barcode_opening.shape[:2] # Shape of the oriented binarized image without numbers

    # Lists of top and bottom black pixels
    list_up = []
    list_down = []

    for x in range(width):
        up = False
        down = False
        for y in range(int(height/2 - 0.1*height)): # Iterate from 0 to half image's height minus 10% of the total height (excluding fake up and down edges, approximately at the middle of the sidebars)
            if barcode_opening[y, x] == 0 and up == False : # Upper pixel has been found
                up = True 
                list_up.append(y) # add the the y coordinate to the list of upper pixels
            
            if barcode_opening[height-y-1, x] == 0 and down == False : # Lower pixel has been found
                down = True
                list_down.append(height - y) # add the the y coordinate to the list of lower pixels (Starts from the bottom to find lower edges)
            if up and down: break

    # Lists without wrong values

    # In certain instances, the bars may not be perfectly straight, and there are black pixels discovered that do not correspond to an edge. Therefore, these incorrect pixels should be excluded from the previous lists

    # List initialization
    list_up_filtered = []
    list_down_filtered = []
    delta = 2 # defines a range to recognize a sequence of at least 3 pixels as an edge
    for u in range(1, len(list_up)-1) :
        if list_up[u-1] - delta < list_up[u] < list_up[u-1] + delta and list_up[u+1] - delta < list_up[u] < list_up[u+1] + delta : # comparing each element of the list_up with respect his left neighbour in the defined range and with respect his right neighbour in the defined range
            list_up_filtered.append(list_up[u]) # if it is in the range , we save it 

    for d in range(1, len(list_down)-1) :
        if list_down[d-1] - delta < list_down[d] < list_down[d-1] + delta and list_down[d+1] - delta < list_down[d] < list_down[d+1] + delta :  # comparing each element of the list_down with respect his left neighbour in the defined range and with respect his right neighbour in the defined range
            list_down_filtered.append(list_down[d]) # if it is in the range , we save it

    up = np.max(list_up_filtered) # the highest y coordinate of the pixel of the shortest bar
    down = np.min(list_down_filtered) # the lowest y coordinate of the pixel of the shortest bar

    # Finding left and right bars

    # Lists of left and right black pixels
    list_left = []
    list_right = []

    # Starting points for x iterations that are updated if a "wrong" bar is found
    left_bound = 0 
    right_bound = width-1

    while(True) :
        for y in range(up , down):   # iteration on the y axis from up to down
            for x in range(left_bound, width):   # for each x we find the far left pixel
                if barcode_opening[y, x] == 0 :
                    list_left.append(x) # add the x coordinate to the list
                    break
        
        std_left = np.sqrt(np.var(list_left)) # computation of the standard deviation of the far left x coordinates
        
        if(std_left > 1) :  # found a "wrong" bar (when the list of the far left pixels has the std > 1, the element found is not a straight bar)
            skip_list = []   # list used to save the thicknesses of the "wrong" bar
            for y in range(up, down):
                skip = 0 # thickness value of the "wrong" bar for a given y coordinate  
                for x in range(min(list_left), width) : # start from the far left pixel of the "wrong" bar
                    if barcode_opening[y, x] == 0 : # if the pixel is black
                        skip += 1 # increment of one the thickness
                    elif skip != 0 : # If we have found the total thickness of the "wrong" bar for a given y
                        skip_list.append(skip) # save the value in the list
                        break
            # Updating of left_bound adding the maximum thickness of the "wrong" bar to the x coordinate of the far left pixel found
            left_bound = min(list_left) + max(skip_list)  # next x cicle skips the "wrong" bar 
            list_left = [] # reset the list for the next iteration
        else :
            break

        if left_bound > width/2:
            print("ERROR: left bar doesn't found..")
            exit()

    while(True) :
        for y in range(up, down):   # iteration on the y axis from up to down
            for x in range(right_bound, 0, -1):   # for each x we find the far right pixel
                if barcode_opening[y, x] == 0 :
                    list_right.append(x) # add the x coordinate to the list
                    break
        
        std_right = np.sqrt(np.var(list_right)) # computation of the standard deviation of the far right x coordinates

        if(std_right > 1) : # found a "wrong" bar (when the list of the far right pixels has the std > 1, the element found is not a straight bar)
            skip_list = []     # list used to save the thicknesses of the "wrong" bar
            for y in range(up, down):
                skip = 0    # thickness value of the "wrong" bar for a given y coordinate   
                for x in range(max(list_right), 0, -1) :   # start from the far right pixel of the "wrong" bar
                    if barcode_opening[y, x] == 0 : # if the pixel is black
                        skip += 1 # increment of one the thickness
                    elif skip != 0 : # If we have found the total thickness of the "wrong" bar for a given y
                        skip_list.append(skip) # save the value in the list
                        break
            # Updating of right_bound adding the maximum thickness of the "wrong" bar to the x coordinate of the far right pixel found
            right_bound = max(list_right) - max(skip_list)  # next x cicle skips the "wrong" bar
            list_right = [] # reset the list for the next iteration
        else :
            break

        if right_bound < width/2:
            print("ERROR: right bar doesn't found...")
            exit()

    left = min(list_left) # the left edge is the minimum of all the x coordinates found 
    right = max(list_right) # the right edge is the maximum of all the x coordinates found 

    # Visualization of the area where computing the parameters as a Green Box 
    result_image = cv2.cvtColor(barcode_opening, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result_image, (left, up), (right , down) , (0,255,0), 2)

    axs[ind,j].imshow(result_image, cmap = 'gray')
    axs[ind,j].set_title(element)
    axs[ind,j].set_xticks([])
    axs[ind,j].set_yticks([])
    j += not (ind + 1) % 7
    ind = (ind + 1) % 7 


    # finding thickness min in order to obtain the X-dimension

    # list initialization
    thickness_list = [] # list of minimum thicknesses for a given y 
    thickness = 0 # Thickness variable as accumulator  
    is_a_bar = False # If we are iterating coordinates inside a bar

    for y in range(up,down): # from up to down
        minimum = width  # Minimum thickness
        for x in range(left,right): # from left to right, we are looking for bars
            if barcode_opening[y, x] == 255 and is_a_bar == True : # if we are outside a bar
                is_a_bar = False # update the variable
                minimum = min(minimum,thickness) # save the new thickness if it is the minimum
                thickness = 0
                    
            if barcode_opening[y, x] == 0 and is_a_bar == False : # if a bar si found
                is_a_bar = True # update the variable
                
            if barcode_opening[y, x] == 0 and is_a_bar == True : # if we find a black pixel inside a bar
                thickness+=1 # increment the thickness
        if minimum > 0: thickness_list.append(minimum) # if thickness is different from 0, save it to the list

    thickness_min = np.mean(thickness_list).astype(int) # compute the mean across all the minimum thickness

    # Image crop - We are considering the barcode ROI plus ten times the minimum thickness of a bar on the left and right sides, and only one time the thickness on the top and bottom sides
    barcode_crop = img_oriented[up + offset_y - thickness_min : down + offset_y + thickness_min, left + offset_x - thickness_min*10 : right + offset_x + thickness_min*10]

    height, width = barcode_crop.shape # Shape of the image
    delta = int(height / 11) # Distance between the scan lines - It's computed such that the scan lines are equally spaced between themselves and the top and bottom edges of the image
    th = [] # Threshold

    y = delta # Set y to compute the firt scan line
    scan_line = np.zeros([10, width]) # Scan lines vectors initialization - It's a matrix with ten rows and as many columns as the width value

    for i in range(10) : # For each scan line
        for x in range(width) : scan_line[i,x] = (barcode_crop[y, x]/255)*100 # Compute the intensity of the pixels as a percentage
        y = y + delta  # Updating y to the next scan line

    # Parameters dictionary, each element is a list where we store the values of that parameter for the ten scan lines
    parameters = {"R_min": [],"R_max": [], "ECmin": [], "modulation": [] , "defects": [], "ERN_max": [], "symbol_contrast": []}

    # Empty list for storing edges - It's a list of ten lists, where each sublist contains the computed edges for each scan line
    edges_list = []

    # Parameters Computation

    # Reflectance
    for i in range(10) : # for each scan line
        R_min = 100 # Minimum reflectance - initialized to the maximum possible value
        R_max = 0 # Maximum reflectance - initialized to the minimum possible value
        for x in range(width):
            R_min = min(R_min, scan_line[i,x]) # The minimum value in the whole scan line
            R_max = max(R_max, scan_line[i,x]) # The maximum value in the whole scan line
        # add values to the list
        parameters["R_min"].append(R_min)
        parameters["R_max"].append(R_max)

    # Symbol contrast computation
    for i in range(10) :
        parameters["symbol_contrast"].append(parameters["R_max"][i] - parameters["R_min"][i]) # Rmax - R_min
    
    # Compute the threshold
    for i in range(10):
        th.append(parameters["symbol_contrast"][i]/2 + parameters["R_min"][i]) # Threshold

    # Edges
    for i in range(10) : # for each scan line
        edges = [] # supporting temporary list for the considered scan line
        for x in range(width-1):
            if scan_line[i,x] > th[i] and scan_line[i,x+1] <= th[i] : # If we cross the threshold and the intensity is decreasing
                edges.append(x) # add the the edge to the list
                    
            if scan_line[i,x] < th[i] and scan_line[i,x+1] >= th[i] : # If we cross the threshold and the intensity is increasing
                edges.append(x) # add the the edge to the list
        edges_list.append(edges) # add the the edges list for a single scan line to the edges_list
            
    # Computation of the space and bar sizes considering only the fifth scan line
    bars_and_spaces_list = [] # Empty list for storing bars and spaces' sizes - It's a list of ten lists, where each sublist contains the computed values
    previous = edges_list[4][0] # Set previous as the first edge

    for x in edges_list[4]: # for all the edges in the fifth scan line
        if x != edges_list[4][0]: bars_and_spaces_list.append((x-previous)/ thickness_min) # compute the difference between consecutive edges, scaled by thickness_min
        previous = x # Set previous as the previous edge

    # ECmin - Minimum edge contrast is the minimum difference between the minimum reflectance in a space and the maximum reflectance in a bar, over pairs of adjacent spaces-bars
    min_v = 100 # supporting temporary variable for the minimum reflectance
    max_v = 0  # supporting temporary variable for the maximum reflectance
    for i in range(10) :  # for each scan line
        ECmin = 100 # initialization
        for x in range(width-1):
            if x in edges_list[i][1:]: # When an edge occurs
                ECmin = min(ECmin, (np.abs(max_v - min_v))) # Compute the ECmin of the previous reflectance difference
                if scan_line[i,x] > th[i] and scan_line[i,x+1] <= th[i] : # If you are going under the threshold
                    min_v = 100 # reset the min
                if scan_line[i,x] < th[i] and scan_line[i,x+1] >= th[i] : # If you are going over the threshold
                    max_v = 0 # Reset the max
            else:
                if scan_line[i,x] > th[i]: # If you are over the threshold
                    max_v = max(max_v, scan_line[i,x]) # Save the max value
                if scan_line[i,x] < th[i]:  # If you are under the threshold
                    min_v = min(min_v, scan_line[i,x]) # Save the minimum value
            
        parameters["ECmin"].append(ECmin) # Save ECmin in the list of parameters for the current scan line

    # Modulation computation
    for i in range(10) :
        parameters["modulation"].append(parameters["ECmin"][i] / parameters["symbol_contrast"][i] * 100) # ECmin / symbol_contrast * 100 

    # ERN max - ERN is defined as the difference between the highest peak and the lowest valley within an element
    for i in range(10) :  # for each scan line
        ERN_max = 0 # initialization
        peak = 0 # supporting temporary variable for a peak
        valley = 100 # supporting temporary variable for a valley
        for x in range(1, width-1):
            if x in edges_list[i]: # When an edge occurs
                if peak != 0 and valley != 100: # If a peak and a valley have been found
                    ern = peak - valley # Compute ern for the previous element
                    ERN_max = max(ERN_max,ern) # Save ERN_max as the maximum ern
                peak = 0 # reset
                valley = 100 # reset
            else:
                if scan_line[i, x-1] < scan_line[i,x] > scan_line[i, x+1]: # If the current intensity is a local maximum
                    peak = max(peak, scan_line[i,x]) # save it as peak if it is the maximum found value
                if scan_line[i, x-1] > scan_line[i,x] < scan_line[i, x+1]: # If the current intensity is a local minimum
                    valley = min(valley, scan_line[i,x]) # save it as valley if it is the minimum found value
        parameters["ERN_max"].append(ERN_max) # Save ERN_max in the parameters list for the current scan line
        

    # Defects computation - it is tha ratio between ERN_max and Symbol Contrast
    for i in range(10) : parameters["defects"].append(parameters["ERN_max"][i] / (parameters["symbol_contrast"][i]) * 100) # ERN_max / symbol_contrast * 100 

    # Marks

    # Marks dictionary, each element is a list where we store the mark of that parameter for the ten scan lines
    marks = {"R_min": [], "ECmin": [], "modulation": [], "defects": [], "symbol_contrast": [], "decodability" : [], "decode" : []}

    # Dictionary as database for the Auxiliary Excel table computed for each image
    image_table = {
        "Rmin" : [],
        "Symbol Contrast" : [],
        "Modulation":[],
        "Defects": [],
        "Number of edges": []
    }

    # Assigning the marks
    for i in range (10):
        if parameters["R_min"][i] > 0.5*parameters["R_max"][i]: marks["R_min"].append(0)
        else: marks["R_min"].append(4)

        if parameters["ECmin"][i] >= 15: marks["ECmin"].append(4)
        else: marks["ECmin"].append(0)

        if parameters["symbol_contrast"][i] >= 70: marks["symbol_contrast"].append(4)
        elif parameters["symbol_contrast"][i] >= 55: marks["symbol_contrast"].append(3)
        elif parameters["symbol_contrast"][i] >= 40: marks["symbol_contrast"].append(2)
        elif parameters["symbol_contrast"][i] >= 20: marks["symbol_contrast"].append(1)
        else: marks["symbol_contrast"].append(0)

        if parameters["modulation"][i] >= 70: marks["modulation"].append(4)
        elif parameters["modulation"][i] >= 60: marks["modulation"].append(3)
        elif parameters["modulation"][i] >= 50: marks["modulation"].append(2)
        elif parameters["modulation"][i] >= 40: marks["modulation"].append(1)
        else: marks["modulation"].append(0)

        if parameters["defects"][i] <= 15: marks["defects"].append(4)
        elif parameters["defects"][i] <= 20: marks["defects"].append(3)
        elif parameters["defects"][i] <= 25: marks["defects"].append(2)
        elif parameters["defects"][i] <= 30: marks["defects"].append(1)
        else: marks["defects"].append(0)

        # Decode and Decodability has always value 4
        marks["decode"].append(4)
        marks["decodability"].append(4)

    min_marks = [min(marks[x][i] for x in marks) for i in range(10)] # List of ten elements with the mimimum mark for each line

    numerical_grade = np.mean(min_marks) # Mean value of the marks - one element for each image

    # Assigning the overall symbol mark
    if  numerical_grade >= 3.5: overall_grade = 'A'
    elif numerical_grade >= 2.5: overall_grade = 'B'
    elif numerical_grade >= 1.5: overall_grade = 'C'
    elif numerical_grade >= 0.5: overall_grade = 'D'
    else : overall_grade = 'F'

    print("{} The overall grade is: {}" .format(element, overall_grade))

    # Assigning data to the table

    # Corners of the Barcode
    left_up = (offset_y + up, offset_x + left)
    left_down = (offset_y + down, offset_x + left)
    right_up = (offset_y + up, offset_x + right)
    right_down = (offset_y + down, offset_x + right)

    # Center of the Barcode
    center = (int((down-up)/2) + offset_y , int((right - left)/2) + offset_x)

    # Adding elements to the database
    Principal_table["Images"].append(element)
    Principal_table["Overall barcode grade"].append(overall_grade)
    Principal_table["X-dimension"].append(thickness_min)
    Principal_table["Height"].append(down-up)
    Principal_table["Left-up"].append([left_up])
    Principal_table["Left-down"].append([left_down])
    Principal_table["Right-up"].append([right_up])
    Principal_table["Right-down"].append([right_down])
    Principal_table["Center"].append([center])
    Principal_table["Orientation"].append(-theta)

    image_table["Rmin"] = parameters["R_min"]
    image_table["Defects"] = parameters["defects"]
    image_table["Symbol Contrast"] = parameters["symbol_contrast"]
    image_table["Modulation"] = parameters["modulation"]

    

    for i in range(10) : image_table["Number of edges"].append(len(edges_list[i]))

    # Creation of a database with Pandas
    df = pd.DataFrame(image_table) # Auxiliary DataFrame
    Image_df = pd.concat([df, pd.DataFrame({"Sequence of bar and space sizes": bars_and_spaces_list})]) # Auxiliary DataFrame

    # Create an excel writer
    with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:
        Image_df.to_excel(writer, sheet_name=element, index=False)

# Creation of a database with Pandas
principal_df = pd.DataFrame(Principal_table) # Principal DataFrame

# Write to Excel
with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:
    principal_df.to_excel(writer, sheet_name="Principal", index=False)

plt.show()





