import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import re
import matplotlib.cm as cm

# <markdowncell>
# ### Helper functions
# <codecell>
def conv2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_2_images(img, img2, title, title2, gray=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.tight_layout()
    if (gray):
        ax1.imshow(img, cmap='gray')
        ax2.imshow(img2,cmap='gray')
    else:
        ax1.imshow(conv2RGB(img))
        ax2.imshow(conv2RGB(img2))
   
    ax1.set_title(title, fontsize=15)    
    ax2.set_title(title2, fontsize=15)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
def show_3_gray_images(img, img2, img3, title, title2, title3):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img, cmap='gray')
    ax1.set_title(title, fontsize=15)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2, fontsize=15)
    ax3.imshow(img3, cmap='gray')
    ax3.set_title(title3, fontsize=15)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# <markdowncell>
# ### Returns the undistorted image
# <codecell>
def collect_calibration_points(images):
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)  
    
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
        # If corners are found, add object points, image points
        if ret == True:        
            imgpoints.append(corners)
            objpoints.append(objp)   
            
    return imgpoints, objpoints

# <markdowncell>
# ### Image undistortion
# <codecell>

def cal_undistort(img, objpoints, imgpoints):  
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
  
# <markdowncell>
# ### Performs perspective transform
# <codecell>
def perspective_transform(img): 
    height =  img.shape[0]
    y_level = 0.64 * height
    top_offset = 100
    side_offset = 250
    src = np.float32([[575, y_level],[710, y_level],[1130, height],[210, height]])
    dst = np.float32([[side_offset, top_offset], [1280-side_offset, top_offset],
                      [1280-side_offset, height],[side_offset, height]])
        
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv

# <markdowncell>
# ### Threshold functions for Sobel gradient
# <codecell>
# Calculate directional gradient
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient
    if orient == 'x':
    	sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
    	sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute values
    sobel = np.absolute(sobel)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return binary_output

# Calculate gradient magnitude
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    return binary_output

# Calculate gradient direction
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.absolute(np.arctan(sobely/sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir > thresh[0]) & (absgraddir < thresh[1])] = 1
    return dir_binary

# <markdowncell>
# ### Combine Sobel gradient filter and color filter
# <codecell>
def apply_filters(img, show_intermediate_results=False):
    img = np.copy(img)       
   
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gaussian_blur(gray, 7)
    # Define sobel kernel size
    ksize = 7
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(30, 255))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(150, 255))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
    # Combine all the thresholding information
    gradient_combined = np.zeros_like(gradx)
    gradient_combined[( (mag_binary == 1) & (gradx == 1) & (dir_binary ==1)) ] = 1
    
    if (show_intermediate_results):
        show_3_gray_images(gray, gradx, grady, 'Gray image', 'Gradient x direction', 'Gradient y direction')
        show_3_gray_images(mag_binary, dir_binary, gradient_combined, 'Gradient magnitude', 'Gradient direction 0.7-1.3', 'Gradients combined binary')
    
    # Convert to HSL color space and separate L and S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    if (show_intermediate_results):
        show_3_gray_images(h_channel, s_channel, l_channel, 'H channel', 'S channel', 'L channel')
       
    h_thresh=(15, 45)
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    # Threshold color channels
    s_thresh=(150, 255)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
   
    l_thresh=(240, 255)
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
   
     
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gray)
    combined_binary[(gradient_combined == 1) | ((s_binary == 1) & (h_binary == 1)) | (l_binary == 1) ] = 1
  
    if (show_intermediate_results):
        show_3_gray_images(h_binary, s_binary, l_binary, 'H channel binary', 'S channel binary', 'L channel binary')
    
    return  combined_binary
    
# <markdowncell>
# ### Undistort and warp image
# <codecell>
    
def warp_binary(img, show_intermediate_results = False, save_out_images=False, img_name="Img"):   
#    cv2.polylines(img, np.int32([src]), True, color=(0,0,255), thickness=3)    
    undist_img = cal_undistort(img, objpoints, imgpoints)
    if (show_intermediate_results):
        show_2_images(img, undist_img, 'Original Image', 'Undistorted Image')
        
    filtered_img_binary = apply_filters(undist_img, show_intermediate_results) 
    
    warped_img, M, Minv = perspective_transform(undist_img)
    if (show_intermediate_results):        
        show_2_images(img, warped_img, 'Undistorted Image', 'Undistorted and Warped Image')
    
    warped_binary_img, M, Minv = perspective_transform(filtered_img_binary)
    if (show_intermediate_results):
        show_2_images(filtered_img_binary, warped_binary_img, 'Binary Image', 'Warped Binary Image', gray=True)

#    warped_binary_img = apply_filters(warped_img, show_intermediate_results)
#    if (show_intermediate_results):
#        show_2_images(warped_img, warped_binary_img, 'Warped Image', 'Warped Filtered Image', gray=True)

    if (save_out_images):
        img_name = re.split("[\\\/.]+",img_name)[-2]
       
        cv2.imwrite("output_images/"+img_name+"_undistorted.jpg",conv2RGB(undist_img))
        cv2.imwrite("output_images/"+img_name+"_undistorted_and_warped.jpg",conv2RGB(warped_img))
        plt.imsave("output_images/"+img_name+"_binary.jpg", np.array(filtered_img_binary), cmap=cm.gray)
        plt.imsave("output_images/"+img_name+"_warped_binary.jpg", np.array(warped_binary_img), cmap=cm.gray)
      
    return warped_binary_img, M, Minv

# <markdowncell>
# ### Find coefficients and fit poly line to lanes
# <codecell>

def fit_lanes(binary_warped, show_intermediate_results=False):
    global initial_fit_done
    global left_fit
    global right_fit
    
    if (show_intermediate_results):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        f.tight_layout()
        ax1.imshow(img)
        ax2.imshow(binary_warped,cmap='gray')
        ax1.set_title('img', fontsize=15)    
        ax2.set_title('binary_warped', fontsize=15)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
       
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    if (show_intermediate_results):
        fig = plt.figure(figsize = (6,4))
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    
    initial_fit_done = True
    
#    initialize_averaged_line_params(lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx)
    
    return ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx

# <markdowncell>
# ### Find coefficients and fit poly line to lanes - based on previously found line fits
# <codecell>

def fit_lanes2(binary_warped, show_intermediate_results=False):  
    global initial_fit_done
    global left_fit
    global right_fit
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    if (show_intermediate_results):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    
    # If the fits seem to be OK, put current values to averaging variables
#    if sanity_check(left_fit, right_fit):
#        lines_averaging(lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx)

    return ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx

# <markdowncell>
# ### Line averaging operations
# <codecell>

def get_running_average(avg_val, cur_val):
    param1 = 0.9
    param2 = 0.1
#    if (avg_val.shape!=cur_val.shape):
#        min_len = min(avg_val.shape[0], cur_val.shape[0])
#        avg_val = avg_val[:min_len]
#        cur_val = cur_val[:min_len]
        
    return avg_val * param1 + cur_val * param2
#
#def initialize_averaged_line_params(lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx):
#    global lefty_avg
#    global righty_avg
#    global leftx_avg
#    global rightx_avg
#    global left_fit_avg
#    global right_fit_avg 
#    global left_fitx_avg 
#    global right_fitx_avg
#    
#    lefty_avg = lefty
#    righty_avg = righty
#    leftx_avg = leftx
#    rightx_avg = rightx
#    left_fit_avg = left_fit
#    right_fit_avg = right_fit
#    left_fitx_avg = left_fitx
#    right_fitx_avg = right_fitx    
#    
#def lines_averaging(lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx):
#    global lefty_avg
#    global righty_avg
#    global leftx_avg
#    global rightx_avg
#    global left_fit_avg
#    global right_fit_avg 
#    global left_fitx_avg 
#    global right_fitx_avg
#    
#    lefty_avg = get_running_average(lefty_avg, lefty)
#    righty_avg = get_running_average(righty_avg, righty)
#    leftx_avg = get_running_average(leftx_avg, leftx)
#    rightx_avg = get_running_average(rightx_avg, rightx)
#    left_fit_avg = get_running_average(left_fit_avg, left_fit)
#    right_fit_avg = get_running_average(right_fit_avg, right_fit)
#    left_fitx_avg = get_running_average(left_fitx_avg, left_fitx,)
#    right_fitx_avg = get_running_average(right_fitx_avg, right_fitx)
# 
#def get_averaged_line_params():
#    global lefty_avg
#    global righty_avg
#    global leftx_avg
#    global rightx_avg
#    global left_fit_avg
#    global right_fit_avg 
#    global left_fitx_avg 
#    global right_fitx_avg
#    
#    return (lefty_avg, righty_avg, leftx_avg, rightx_avg, left_fit_avg, right_fit_avg, left_fitx_avg, right_fitx_avg)
    
    
# <markdowncell>
# ### Initialize global variables
# <codecell>

#global initial_fit_done
#global left_fit
#global right_fit
#
#global lefty_avg
#global righty_avg
#global leftx_avg
#global rightx_avg
#global left_fit_avg
#global right_fit_avg 
#global left_fitx_avg 
#global right_fitx_avg

initial_fit_done=False
left_fit=0.0
right_fit=0.0
curvature_avg = 0

def initialize_global_vars():
    global initial_fit_done
    global curvature_avg
    initial_fit_done = False
    curvature_avg = 0
    
# <markdowncell>
# ### Check whether the first coefficients of left and right fit are of the same sign
# <codecell>
def sanity_check(left_fit, right_fit):
    if (left_fit[0]*right_fit[0]<0):
        return False
    else:
        return True
    
# <markdowncell>
# ### Find real curvature and center of the car, return final output
# <codecell>
def calc_curvature_and_return_final_img(img, binary_warped, ploty, lefty, righty, leftx, rightx, 
                   left_fit, right_fit, left_fitx, right_fitx, Minv, show_intermediate_results=False):
    global curvature_avg
    height = img.shape[0]
    ym_per_pix = 30/height # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    
    rightx_int = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    leftx_int = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    center = abs(640 - ((rightx_int+leftx_int)/2))

# Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    pts = np.hstack((pts_left, pts_right))
    #cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0]))
    result = cv2.addWeighted(img, 1, newwarp, 0.5, 0)
    
    curvature = int((left_curverad + right_curverad)/2)
    if (curvature_avg==0):
        curvature_avg = curvature
    else:
        curvature_avg = int(get_running_average(curvature_avg, curvature))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    #text_radius = 'Radius of curvature is {}m'.format(int(min(left_curverad, right_curverad)))
    text_radius = 'Radius of curvature is {}m'.format(curvature_avg)
    cv2.putText(result,text_radius,(200,100), font, 1,(255,255,255),2)
    
#    cv2.putText(result,'Left: {}m, right {}m'.format(left_curverad, right_curverad),(200,250), font, 1,(255,255,255),2)
    
    
    
    if center < 640:
        text_position = 'Vehicle is {:.2f}m left of center'.format(center*3.7/700)
    else:
        text_position = 'Vehicle is {:.2f}m right of center'.format(center*3.7/700)
      
    cv2.putText(result,text_position,(200,175), font, 1,(255,255,255),2)
        
    if (show_intermediate_results):
        fig = plt.figure(figsize = (6,3))
        plt.imshow((result))
        plt.text(200, 100, text_position,
        style='italic', color='white', fontsize=10)       
        plt.text(200, 175, text_radius, style='italic', color='white', fontsize=10)
      
   

    return result
    
# <markdowncell>
# ### Main pipeline
# <codecell>       
count = 1

def process_pipeline(img):
    global initial_fit_done
    global left_fit
    global right_fit
    global count
   
    warp_binary_img, M, Minv = warp_binary(img)
#    if (initial_fit_done==False):
    ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx = fit_lanes(warp_binary_img)
#    else:
#        ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx = fit_lanes2(warp_binary_img)


#        ploty = fit_lanes2(warp_binary_img)
    
#    lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx = get_averaged_line_params()
    
    result = calc_curvature_and_return_final_img(img, warp_binary_img, ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, Minv)
    count = count + 1
    
    return result
    

# <markdowncell>
# ### Client code starts here
# Obtain cal points for camera calibration
# <codecell>

cal_images = glob.glob('camera_cal/calibration*.jpg')
imgpoints, objpoints = collect_calibration_points(cal_images)

test_image = 'camera_cal/calibration5.jpg'
img = cv2.imread(test_image)
img = conv2RGB(img)
undist_img = cal_undistort(img, objpoints, imgpoints)
show_2_images(img, undist_img, "Original image", "Undistorted image")

# <markdowncell>
# ### Process each image in test_images folder - save results
# <codecell>

for image_name in glob.glob('test_images/test*.jpg'):
    img = cv2.imread(image_name)
    img=conv2RGB(img)
    warp_binary_img, M, Minv = warp_binary(img, False, True, image_name)
    ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx = fit_lanes(warp_binary_img, True)
    final_image = calc_curvature_and_return_final_img(img, warp_binary_img, ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, Minv, False)
    
    image_name = re.split("[\\\/.]+",image_name)[-2]
    cv2.imwrite("output_images/"+image_name+"_final.jpg",conv2RGB(final_image))
    
    fig = plt.figure(figsize = (8,4))
    plt.imshow(final_image)

# <markdowncell>
# ### Process one test image
# <codecell>
img = cv2.imread('test_images/test_video3.png')
img_processed = process_pipeline(img)
fig = plt.figure(figsize = (8,4))
plt.imshow(img_processed)


# <markdowncell>
# ### Process one test image
# <codecell>
initialize_global_vars()


for image_name in glob.glob('test_images/test_video*.png'):
    img = cv2.imread(image_name)
    img=conv2RGB(img)
    img_processed = process_pipeline(img)
    fig = plt.figure(figsize = (8,4))
    plt.imshow(img_processed) 

# <markdowncell>
# ### Process one test image
# <codecell>
image_name='test_images/straight_lines1.jpg'
#image_name='test_images/test_from_vid11.png'
image_name='test_images/test5.jpg'
image_name='test_images/test_video19.png'
img = cv2.imread(image_name)
img=conv2RGB(img)
warp_binary_img, M, Minv = warp_binary(img, True, False, image_name)

ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx = fit_lanes(warp_binary_img, True)

#ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx  = fit_lanes2(warp_binary_img)
#lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx = get_averaged_line_params()

final_image = calc_curvature_and_return_final_img(img, warp_binary_img, ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, Minv, False)

fig = plt.figure(figsize = (8,4))
plt.imshow(final_image)
#    
    
# <markdowncell>
# ### Process video stream
# <codecell>
### Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#from IPython import get_ipython
# Set up lines for left and right
initialize_global_vars()
video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_pipeline) #NOTE: this function expects color images!!

#get_ipython().magic('time white_clip.write_videofile(video_output, audio=False)')
white_clip.write_videofile(video_output, audio=False)
        
    