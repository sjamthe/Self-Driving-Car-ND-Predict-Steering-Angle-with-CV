
# coding: utf-8

# # **Finding Lane Lines on the Road**

# In[6]:

#importing some useful packages
import sys
import os
#add syspath for cv2under diff envornment
sys.path.append('/Users/sjamthe/anaconda/envs/condapy3/lib//python3.5/site-packages')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from scipy import stats
#get_ipython().magic('matplotlib inline')


# In[7]:

def show_image(title,img,cmap=None):
    global debug

    #return
    if(debug==1):
        print (title)
        fig = plt.figure()
        plt.title(title)
        plt.imshow(img,cmap)
        plt.show()
        plt.close(fig)

def yellowgrayscale(image):
    #enhance yellow then find grayscale

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of yellow color in HSV
    #lower = np.array([40,40,40])
    #upper = np.array([150,255,255])

    #RGB limits
    lower = np.array([80,80,40])
    upper = np.array([255,255,80])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(image, lower, upper)
    #show_image('mask',mask)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    res = cv2.addWeighted(res, 1.0, image, 1.0, 0)
    res = grayscale(res)

    return res

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def median_blur(img, kernel_size):
    """Applies a median_blur Noise kernel"""
    return cv2.medianBlur(img, kernel_size)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

    if lines is None:
        return

    for line in lines:
        for x1,y1,x2,y2 in line:
            #print (x1,y1,x2,y2)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# In[8]:

def test_lines(img, lines,title="test_lines"):
    global debug
    global main_image

    if(debug == 1):
        line_img = np.zeros(img.shape, dtype=np.uint8)
        #print ("size=",lines.size)
        draw_lines(line_img, lines,thickness=2)
        combined_img = mark_lanes(line_img, main_image)
        show_image(title,combined_img)

# In[133]:

def top(img):
    """
        Define where to truncate the image. This function brings hard-coding to one place.
    """
    hardcoding = 0.4

    image_height = img.shape[0]
    image_width = img.shape[1]
    image_top = hardcoding*image_height #top is the lowest top x value of region on interest
    image_bottom = 0.8*image_height

    return [image_top,image_bottom]

#my modified and new functions
def lane_lines(img, lines):
    """
    `lines` should be the output of a cv2.HoughLinesP.

    Returns left lane and right lane in a list.
    """
    global debug
    global image_int

    #if(lines is Null):
     #   print ("Error: No Houghlines detected for image ",image_cnt)

     if(lines is None):
        return None

    num_lines = lines.shape[0]
    #rlines & llines store y values corresponding top and bottom x values along with weight (length of hf line y2-y1)
    llines = []
    rlines = []
    image_height = img.shape[0]
    image_width = img.shape[1]
    [image_top,image_bottom] = top(img) #top is the lowest top x value of region on interest

    #for test drawing of what lines we are selecting
    tl_lines = []
    tr_lines = []
    skip_lines = []

    #filter out all the points that have slope outside 80-100 degrees
    for line in lines:
        for x1,y1,x2,y2 in line:
            skipped=True
            if(x2 == x1):
                continue #skip vertical lines
            m = np.round((y2-y1)/(x2-x1),1) #round to 1/10th
            b = np.round(y2 - m*x2,0) #round to integer
            if(debug):
                print("point (x1,y1,x2,y2) ",x1,y1,x2,y2," m,b",m,b,"image_width=",image_width)
            #ignore high slopes and intercept that are not in the image
            if(m >= 0.1 and x1 > .4*image_width/2):
            #if(x1 > 0.3*image_width and  m >.1 and b < 0.5*image_width and b >= -100):
                #find where the line intercepts the image bottom,
                x_bottom = int((image_bottom -b)/m/10)*10 #round it to 10 pixels
                #find where the line intercepts the top of region of interest,
                x_top = int((image_top -b)/m/10)*10 #round it to 10 pixels
                #if(x_bottom > 0.7*image_width and x_bottom < image_width):
                #added x_top to prevent horizontal lines in middle of image!
                #could we have used x1 > .5*imagewidth?
                if(x_bottom > 250 and x_top > 0):
                    newline = [x_bottom,x_top,(y2-y1),y2]
                    skipped=False
                    if(debug):
                        print("right point added [x_bottom,x_top,(y1-y2),y1]", [x_bottom,x_top,(y1-y2),y1])
                    rlines.append(newline)
                    tr_lines.append(line)
                    test_lines(img,[line],title=[m,b,x1,x2])
                else:
                    if(debug):
                        print("skipping right point x_bottom(",x_bottom,
                            ") not between 0.7*image_width and  image_width",image_width)
            elif(m<=-.1):
                 if(b <= image_width):
                    #find where the line intercepts the image bottom,
                    x_bottom = int((image_bottom -b)/m/10)*10 #round it to 10 pixels
                    #find where the line intercepts the top of region of interest,
                    x_top = int((image_top -b)/m/10)*10 #round it to 10 pixels
                    if(x_bottom < 120 and x_bottom > -160 and x_top < 250):
                        newline = [x_bottom,x_top,(y1-y2),y1]
                        llines.append(newline)
                        tl_lines.append(line)
                        skipped=False
                        if(debug):
                            print("left point added [x_bottom,x_top,(y1-y2),y1]", [x_bottom,x_top,(y1-y2),y1])
                        test_lines(img,[line],title=[m,b,x1,x2])
                    else:
                        if(debug):
                            print("left point skipped NOT (x_bottom < 100 and x_bottom > -160) :",x_bottom)
            else:
                if(debug):
                    print("point skipped due to slope,intercept or x2 extending image_width",m,b,x2,image_width)
            if(skipped):
                skip_lines.append(line)
                #print("skipping (x1,y1,x2,y2) ",x1,y1,x2,y2," slope,intercept",m,b)

    #test_lines(img,skip_lines)
    new_lines = np.array([np.array(llines),np.array(rlines)])
    #print ("new_lines shape=",new_lines.shape)
    #print ("shapes of llines,rlines=",new_lines[0].shape,new_lines[1].shape)
    return new_lines

def lines_to_lane(img, lines, side):
    """
    convert lines to single lane based on their weights (length of lines and closeness to car)
    """
    global debug
    global image_cnt

    [top_y, bottom_y] = top(img)
    if(len(lines) == 0):
        print("no lane found for image_cnt=",image_cnt)
        #lane = np.array([0, 0, 0, 0], dtype=np.int32)
        lane = None
        return lane

    #sort the lines, also we prefer lines closer to car.
    # how about taking the top two?
    if(side == 'left'):
        #print("left before",lines)
        lines = sorted(lines,key=lambda x: -x[0])[:2]
        #print("left after",lines)
        #lines = np.sort(lines,axis=0)[::-1][:2]
    else:
        #print("right before",lines)
        lines = sorted(lines,key=lambda x: x[0])[:2]
        #print("right after",lines)

    #print (lines.size)
    #print (lines)
    #print (np.median(lines,axis=0),np.std(lines,axis=0))
    median_bottom = np.median(lines,axis=0)[0]
    std_bottom = np.std(lines,axis=0)[0]
    median_top = np.median(lines,axis=0)[1]
    std_top = np.std(lines,axis=0)[1]
    mean_length = np.mean(lines,axis=0)[2]
    mean_proximity = np.mean(lines,axis=0)[3]
    if(debug):
        print("median_bottom,std_bottom,",median_bottom,std_bottom)
        print("std_bottom/median_bottom",(std_bottom/median_bottom))
        print("median_top,std_top,",median_top,std_top)
        print("(std_top/median_top)",(std_top/median_top))

    product_sum=0
    bottom_sum=0
    top_sum=0
    count=0
    for line in lines:
        #print ("line=",line)
        [x_bottom,x_top,length,proximity] = line
        #print ("np.abs(x_top-median_top)",np.abs(x_top-median_top))
        #only chose points that are withing 1 std of mean ??? should it be median?
        if( np.abs(x_bottom-median_bottom) > std_bottom or
               np.abs(x_top-median_top) > std_top):
            if(debug):
                print ("skipping (x_bottom,x_top,length,proximity)", x_bottom,x_top,length,proximity)
        else:
            if(debug):
                print ("good (x_bottom,x_top,length,proximity)", x_bottom,x_top,length,proximity)
            #product=(length/mean_length+proximity/mean_proximity+(160-x_bottom)/(160-median_bottom))
            product = length*proximity
            #product=length
            product_sum = product_sum+product
            bottom_xp = product*x_bottom
            bottom_sum = bottom_sum+bottom_xp
            top_xp = product*x_top
            top_sum = top_sum+top_xp
            count=count+1

    if(count>0):
        bottom_x = int(bottom_sum/product_sum)
        top_x = int(top_sum/product_sum)
        lane = np.array([bottom_x, bottom_y, top_x, top_y], dtype=np.int32)
    else:
        #cv2.imwrite("test_images/noleft-"+str(image_cnt)+".jpg",img)
        print("no lane found for image_cnt=",image_cnt)
        #lane = np.array([0, 0, 0, 0], dtype=np.int32)
        lane = None
    if(debug):
        print ("lines_to_lane=",lane)
    return lane

def concat_hist_lines(both_lines):
    """
    ' Look these lines to all hist lines going back n days and return those lines
    / image_cnt is the number of this image
    input is a list with two arrays (left,right). each array has a list of lines
    we should return similar list after concatenating both arrays.
    """
    global all_lines
    global  image_cnt
    global debug

    max_hist = 1
    start = 0

    #print("input=",both_lines)
    #print('input list sizes=',len(both_lines[0]),len(both_lines[1]))

    if(image_cnt == 1):
        all_lines = [both_lines]
    else:
        all_lines.append(both_lines)
        size=len(all_lines)
        #print("len of all_lines = ",size)
        if(size > max_hist):
            start = size-max_hist
            all_lines = all_lines[start:]
    #image_cnt += 1
    #print("len of all_lines after trunc = ",len(all_lines))

    left_list = []
    right_list = []
    for listvalue in all_lines:
        left = listvalue[0]
        for line in left:
            left_list.append(line)

        right = listvalue[1]
        for line in right:
            right_list.append(line)

    #print('output list sizes=',len(left_list),len(right_list))
    if(len(left_list)==0):
        print("sending zero length left line for image_cnt=",image_cnt)
    if(len(right_list)==0):
        print("sending zero length right line for image_cnt=",image_cnt)
    hist_lines = [left_list,right_list]

    #print("output",hist_lines)
    return hist_lines

def are_lanes_ok(lanes):
    global prev_lanes
    global debug
    global image_cnt

    #are we here first time?
    if(len(prev_lanes) == 0):
        prev_lanes = lanes
        return lanes

    [prev_left_lane,prev_right_lane] = prev_lanes
    [left_lane, right_lane] = lanes
    #print(prev_left_lane,left_lane)
    #print(prev_right_lane,right_lane)

    #make sure x co-ordinates are not far apart (100 points)
    allowed=200

    if(left_lane is None):
        print("for image_cnt=", image_cnt,
            "NOT keeping prev left lane: prev, new ",prev_left_lane, left_lane)
        #left_lane = prev_left_lane
    elif (prev_left_lane is not None and abs(left_lane[0]-prev_left_lane[0]) > allowed):
       #or abs(left_lane[2]-prev_left_lane[2]) > allowed):
        #too much shift, keep old lane
        print("for image_cnt=", image_cnt,
            "keeping prev left lane: prev, new ",prev_left_lane, left_lane)
        left_lane = prev_left_lane

        #abs(right_lane[0]-prev_right_lane[0]) > allowed):
    if(right_lane is None):
        print("for image_cnt=", image_cnt,
            "NOT keeping prev right lane: prev, new ", prev_right_lane, right_lane)
        #right_lane = prev_right_lane
    elif(prev_right_lane is not None and abs(right_lane[2]-prev_right_lane[2]) > allowed):
        #too much shift, keep old lane
        print("for image_cnt=", image_cnt,
            "keeping prev right lane: prev, new ", prev_right_lane, right_lane)
        right_lane = prev_right_lane

    #print("prev=",prev_lanes)
    lanes = [left_lane, right_lane]
    print("image = ", image_cnt, "left_lane=",left_lane, "right_lane=",right_lane)
    prev_lanes = lanes

    return lanes

def steering_angle(image, lanes):
    ""
    ' returns -99 if angle cannot be calculated'
    ""
    global image_cnt
    global speed
    global throttle
    global angle

    steer_away = .15
    sharp_turn = .04
    left_low_limit = -20
    left_high_limit = 0
    right_low_limit = 300
    right_high_limit = 320
    safety_throttle = .03
    default_throttle = .08
    speed_limit = 8
    safety_speed = 3
    max_angle = 0.43 #"25 degrees"


    [left_lane, right_lane] = lanes

    if(left_lane is None and right_lane is None):
        print("image = ",image_cnt,"Assert: cannot see both lanes, speed = ",speed,"old angle=",angle)
        if(speed >= safety_speed):
            throttle = -2*default_throttle
        elif(speed <= 1): #we may have stopped reverse dir & angle
            throttle = default_throttle
            angle = -1*angle
        else:
            throttle = safety_throttle
            angle = np.min([max_angle, angle*2])
        #return old angle
        return angle

    if(right_lane is not None):
        [xr1, yr1, xr2, yr2] = right_lane

        if((xr2-xr1) == 0):
            print("image = ",image_cnt,"Assert: right lane x2==x1", xr2,xr1)
            if(speed < safety_speed):
                throttle = safety_throttle
            else:
                throttle = -2*default_throttle
            return -1*steer_away

        mr = (yr2-yr1)/(xr2-xr1)
        br = yr2 - mr*xr2
        print("image = ", image_cnt,
            ", right lane slope = ",np.round(mr,2),", x_bottom = ", xr1)

        #are we too close    to right? turn left
        if(xr1 < right_low_limit):
            print("image = ",image_cnt,"too close to right (",xr1,"<",right_low_limit
                ,"),slope=",mr,". steer to left")
            angle = np.min([steer_away,math.atan(np.abs(mr))/2])
            if(speed < safety_speed):
                throttle = safety_throttle
                angle = np.min([max_angle, angle*2])
            else:
                throttle = -2*default_throttle
            return -1*angle

    if(left_lane is not None):
        [xl1, yl1, xl2, yl2] = left_lane

        if((xl2-xl1) == 0):
            print("image = ",image_cnt,"Assert: left lane x2==x1", xl2,xl1)
            if(speed < safety_speed):
                throttle = safety_throttle
            else:
                throttle = -2*default_throttle
            return steer_away

        ml = (yl2-yl1)/(xl2-xl1)
        bl = yl2 - ml*xl2

        print("image = ", image_cnt,
            ", left lane slope = ",np.round(ml,2),", x_bottom = ", xl1)

        #are we too close to left lane? turn right
        if(xl1 > left_high_limit):
            print("image = ",image_cnt,"Too close to left (",xl1,'>',left_high_limit,
                "),slope = ",ml,". steer to right")
            angle = np.min([steer_away,math.atan(np.abs(ml))/2])
            if(speed < safety_speed):
                throttle = safety_throttle
                angle = np.min([max_angle, angle*2])
            else:
                throttle = -2*default_throttle
            return angle

    if(right_lane is not None and left_lane is not None):
        if((mr - ml) == 0):
            print("image = ",image_cnt,"Assert: mr = ml", mr,ml)
            if(speed < safety_speed):
                throttle = safety_throttle
            else:
                throttle = -2*default_throttle
            angle = 0
            return angle

        x = (bl - br)/(mr - ml)
        y = ml * x + bl

        imshape = image.shape
        image_height=imshape[0]
        image_width=imshape[1]
        center = image_width/2
        # Assume center of car as center of bottom border of image
        angle = math.atan((x-center)/(image_height - y))/4 #try small angles
        angle = np.round(angle,2)
        print ("image = ", image_cnt, "calculated angle = ",angle)

        if(speed >= speed_limit):
            throttle = -2*default_throttle
        elif(np.abs(angle) > sharp_turn and speed >= safety_speed):
            throttle = -2*default_throttle
        elif(np.abs(angle) > sharp_turn and speed < safety_speed):
            throttle = safety_throttle
        elif(speed < speed_limit):
            throttle = default_throttle
        else:
            throttle = -2*default_throttle #When will we come here? never i think
        return angle
    if(right_lane is None):
        if(xl1 > left_low_limit):
            #no need to panic, keep steady
            angle = 0
            if(speed >= speed_limit):
                throttle = -2*default_throttle
            elif(speed <= 1): #we may have stopped reverse dir & angle
                throttle = default_throttle
            else:
                throttle = safety_throttle
            return angle
        else:
            print("image = ",image_cnt,"Missing right_lane,left too far. steer to left with slope",ml)
            angle = np.min([steer_away,math.atan(np.abs(ml))/2])
            if(speed < safety_speed):
                throttle = safety_throttle
                angle = np.min([max_angle, angle*2])
            else:
                throttle = -2*default_throttle
            return angle
    if(left_lane is None):
        if(xr1 < right_high_limit):
            #keep it steady may be we don't need to panic
            angle = 0
            if(speed >= speed_limit):
                throttle = -2*default_throttle
            elif(speed <= 1): #we may have stopped reverse dir & angle
                throttle = default_throttle
            else:
                throttle = safety_throttle
            return angle
        else:
            print("image = ",image_cnt,"Missing left_lane,right too far, steer to right with slope",mr)
            angle = np.min([steer_away,math.atan(np.abs(mr))/2])
            if(speed < safety_speed):
                throttle = safety_throttle
                angle = np.min([max_angle, angle*2])
            else:
                throttle = -2*default_throttle
            return -1*angle

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    lines are drawn between y values of vertices_top and vertices_bottom

    Returns an image with hough lines drawn.
    """
    global angle

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    new_lines = lane_lines(img, lines)

    #here we should store historic lane_lines together
    concat_lines = concat_hist_lines(new_lines)

    left_lane = lines_to_lane(img, concat_lines[0],'left')
    right_lane = lines_to_lane(img, concat_lines[1],'right')

    [left_lane,right_lane] = are_lanes_ok([left_lane,right_lane])
    angle = steering_angle(img, [left_lane,right_lane])


    #we need points for draw lanes array
    line_img = np.zeros(img.shape, dtype=np.uint8)

    if(left_lane is  None and right_lane is  None):
        lanes = None
    elif(left_lane is None):
        lanes = np.array([np.array([right_lane])])
    elif(right_lane is None):
        lanes = np.array([np.array([left_lane])])
    else:
        lanes = np.array([np.array([left_lane]),np.array([right_lane])])

    draw_lines(line_img, lanes,thickness=5)

    return line_img


# In[140]:

def fetch_test_image(folder,i=83):

    #image = mpimg.imread('test_images/'+folder+'/test-'+str(i)+'.jpg')
    image = mpimg.imread('./sample1/IMG/center_2016_12_02_23_17_58_295.jpg')
    return image

def fetch_image(i=0):

    #reading in an image from test
    images = ['test_images/solidWhiteCurve.jpg',
     'test_images/solidWhiteRight.jpg',
     'test_images/solidYellowCurve.jpg',
     'test_images/solidYellowCurve2.jpg',
     'test_images/solidYellowLeft.jpg',
     'test_images/whiteCarLaneSwitch.jpg',
      'test_images/challenge/test-14.jpg']
    if(i > 6):
        i = 6
    if(i<0):
        i=0
    image = mpimg.imread(images[i])

    return image


# In[120]:

def get_houg_lines_image(image):
    """
    reads a color image and returns gray-image with masked region
    """
    # define a region to find the lines
    imshape = image.shape
    image_height=imshape[0]
    image_width=imshape[1]
    vertices_left = 0
    vertices_right = image_width-vertices_left
    vertices_mid = image_width*.5
    [vertices_top,vertices_bottom] = top(image)

    vertices = np.array([[(vertices_left,vertices_bottom),(vertices_left, vertices_top),
                          (vertices_right, vertices_top), (vertices_right,vertices_bottom)
                         ]], dtype=np.int32)
    #print(top(image))
    masked_image = region_of_interest(image, vertices)
    #show_image("masked_image",masked_image)

    #get a grey scale image
    gray = grayscale(masked_image)
    #gray = yellowgrayscale(masked_image)
    #show_image("grayscale",gray,cmap='gray')

    #equalize = cv2.equalizeHist(gray) didnt work
    #show_image("equalize",equalize,cmap='gray')

    #apply blur
    #gray = gaussian_blur(gray,5)
    #show_image("gaussian_blur",gray,cmap='gray')

    gray = median_blur(gray,5)
    show_image("median_blur",gray,cmap='gray')

    #apply the canny transform to get edges
    edges = canny(gray, 50,150)
    #show_image("canny",edges,cmap='gray')

    #Remask to remove horizontal edges near the top
    vertices = np.array([[(vertices_left,vertices_bottom-10),(vertices_left, vertices_top+10),
                          (vertices_right, vertices_top+10), (vertices_right,vertices_bottom-10)
                         ]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)
    show_image("masked_edges",masked_edges,cmap='gray')

    """
    TODO: Do we need to change these constants in the run? (40,100,160)
    """
    rho = 1          # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 40 #15    # minimum number of points on a line (intersections in Hough grid cell)
    min_line_len = 40 #20  # minimum number of pixels making up a line
    max_line_gap = 40 #20  # maximum gap in pixels between connectable line segments

    hough_lines_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    #show_image("hough_lines",hough_lines_img,cmap='gray')


    return hough_lines_img


# In[12]:

def mark_lanes(hough_lines_img, image):
    """
    Takes hough_lines, image as input and returns combined image
    """
    #now let us draw weighted image with hough_lines on top of original image
    #print (hough_lines_img.shape, image.shape)
    blank = np.zeros_like(hough_lines_img)
    color_edges = np.dstack((hough_lines_img, blank, blank))
    combined_image = weighted_img(color_edges,image)
    #show_image("combined_image", combined_image)
    return combined_image


debug=0
image_cnt=0
all_lines = []
prev_lanes = []
prev_angle = 0
angle = 0
main_image = None
throttle = None
speed = None

def cvangle(cnt, image,t=0, s=0):
    global angle
    global prev_angle
    global image_cnt
    global throttle
    global speed

    throttle = t
    speed = s
    image_cnt = cnt
    hough_lines_img = get_houg_lines_image(image)
    combined_image = mark_lanes(hough_lines_img, image)

    if(angle != -99):
        prev_angle = angle
    else:
        angle = 0
    return angle, throttle, combined_image

if __name__ == '__main__':
    dir = sys.argv[1]
    startswith = sys.argv[2]
    if(sys.argv[3] is not None):
        debug = 1

    files = os.listdir(dir)
    files = sorted(files)
    for  afile in files:
        if(afile.startswith(startswith)):
            image_cnt += 1
            image = mpimg.imread(dir + '/' + afile)
            main_image = image
            angle,throttle,combined_image = cvangle(image_cnt, image)

            print("steering_angle = ",angle)
            plt.imshow(combined_image)
            #plt.savefig('./output/'+ afile)
            plt.show()
