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
y_level = 470
src = np.float32([[490, y_level],[810, y_level],[1250, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0],[1250, 720],[40, 720]])

def perspective_transform(img): 
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
    # Define sobel kernel size
    ksize = 7
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(150, 255))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(.7, 1.3))
    # Combine all the thresholding information
    gradient_combined = np.zeros_like(dir_binary)
    gradient_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    if (show_intermediate_results):
        show_3_gray_images(gray, gradx, grady, 'gray', 'gradx', 'grady')
        show_3_gray_images(mag_binary, dir_binary, gradient_combined, 'mag_binary', 'dir_binary', 'gradient_combined')
    
    # Convert to HSL color space and separate L and S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    if (show_intermediate_results):
        show_3_gray_images(h_channel, s_channel, l_channel, 'h_channel', 's_channel', 'l_channel')
       
    # Threshold color channels
    s_thresh=(170, 255)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
   
    h_thresh=(10, 100)
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
     
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gray)
    combined_binary[(gradient_combined == 1) | (s_binary == 1) ] = 1
  
    if (show_intermediate_results):
        show_3_gray_images(h_binary, s_binary, combined_binary, 'h_binary', 's_binary', 'Finally Filtered Image Binary')
    
    return  combined_binary
    

# <markdowncell>
# ### Main code
# <codecell>
    
def warp_binary(img, show_intermediate_results = False, save_out_images=False, img_name="Img"):       
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

    if (save_out_images):
        img_name = re.split("[\\\/.]+",img_name)[-2]
       
        cv2.imwrite("output_images/"+img_name+"_undistorted.jpg",undist_img)
        cv2.imwrite("output_images/"+img_name+"_undistorted_and_warped.jpg",warped_img)
        plt.imsave("output_images/"+img_name+"_binary.jpg", np.array(filtered_img_binary), cmap=cm.gray)
        plt.imsave("output_images/"+img_name+"_warped_binary.jpg", np.array(warped_binary_img), cmap=cm.gray)
      

    return warped_binary_img, M, Minv


# <markdowncell>
# ### Main code
# <codecell>

#img = cv2.imread('test_images/test2.jpg')
#binary_warped = warp_binary(img)

def fit_lanes(binary_warped, show_intermediate_results=False):

    if (show_intermediate_results):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        f.tight_layout()
        ax1.imshow(conv2RGB(img))
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
    
    return ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx
    
# <markdowncell>
# ### Main code
# <codecell>
# Define conversions in x and y from pixels space to meters
def calc_curvature_and_return_final_img(img, binary_warped, ploty, lefty, righty, leftx, rightx, 
                   left_fit, right_fit, left_fitx, right_fitx, Minv, show_intermediate_results=False):
    ym_per_pix = 30/720 # meters per pixel in y dimension
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
    
    rightx_int = right_fit[0]*720**2 + right_fit[1]*720 + right_fit[2]
    leftx_int = left_fit[0]*720**2 + left_fit[1]*720 + left_fit[2]
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
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_radius = 'Radius of curvature is {}m'.format(int(min(left_curverad, right_curverad)))
    cv2.putText(result,text_radius,(200,100), font, 1,(255,255,255),2)
    
    
    if center < 640:
        text_position = 'Vehicle is {:.2f}m left of center'.format(center*3.7/700)
    else:
        text_position = 'Vehicle is {:.2f}m right of center'.format(center*3.7/700)
      
    cv2.putText(result,text_position,(200,175), font, 1,(255,255,255),2)
        
    if (show_intermediate_results):
        fig = plt.figure(figsize = (6,3))
        plt.imshow(conv2RGB(result))        
        plt.text(200, 100, text_position,
        style='italic', color='white', fontsize=10)       
        plt.text(200, 175, text_radius, style='italic', color='white', fontsize=10)
      

    return result
    
# <markdowncell>
# ### Main code
# <codecell>       
def process_pipeline(img):
    warp_binary_img, M, Minv = warp_binary(img)
    ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx = fit_lanes(warp_binary_img)
    return calc_curvature_and_return_final_img(img, warp_binary_img, ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, Minv)
    

# <markdowncell>
# ### Main code
# <codecell>

cal_images = glob.glob('camera_cal/calibration*.jpg')
imgpoints, objpoints = collect_calibration_points(cal_images)


# <markdowncell>
# ### Process each image in test_images folder - save results
# <codecell>

for image_name in glob.glob('test_images/test*.jpg'):
    img = cv2.imread(image_name)
    warp_binary_img, M, Minv = warp_binary(img, False, True, image_name)
    ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx = fit_lanes(warp_binary_img, True)
    final_image = calc_curvature_and_return_final_img(img, warp_binary_img, ploty, lefty, righty, leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, Minv, False)
    
    image_name = re.split("[\\\/.]+",image_name)[-2]
    cv2.imwrite("output_images/"+image_name+"_final.jpg",final_image)
    
    fig = plt.figure(figsize = (8,4))
    plt.imshow(conv2RGB(final_image)) 

# <markdowncell>
# ### Process one test image
# <codecell>
img = cv2.imread('test_images/test2.jpg')
img_processed = process_pipeline(img)
fig = plt.figure(figsize = (8,4))
plt.imshow(conv2RGB(img_processed))        

# <markdowncell>
# ### Main code
# <codecell>
### Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from IPython import get_ipython
# Set up lines for left and right

video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_pipeline) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(video_output, audio=False)')
#%time white_clip.write_videofile(white_output, audio=False)
        