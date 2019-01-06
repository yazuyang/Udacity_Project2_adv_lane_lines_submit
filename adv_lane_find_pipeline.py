import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import tqdm

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 

        self.left_fit  = [0,0,0]
        self.right_fit = [0,0,0]

# get_undistorted image
'''
In:
    img   numpy array   [ysize, xsize, 3]
    mtx   numpya rray   [3,3]    "return parameter of cv2.calibrateCamera"
    dist  numpya rray   [5]      "return parameter of cv2.calibrateCamera"
Out:
    undistorted  numpy array   [ysize, xsize, 3]
'''
def get_undistorted(img, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted

# get binary image
'''
IN:
    img              numpy array   [ysize, xsize, 3]
    s_thresh         tuple         (2)                threshold of s in HLS color space 
    sx_thresh        tuple         (2)                threshold of s in xgradient
Out:
    combined_binary   numpy array   [ysize, xsize, 1] binary image(True:255, False:0)
    color_binary      numpy array   [ysize, xsize, 3]
'''
def get_binary(img, s_thresh=(90, 255), sx_thresh=(60, 255), flag_plot=False):
    
    img = np.copy(img)
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel filter with x direction 
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    # combined x and color binary
    combined_binary = sxbinary | s_binary
    
    # plot
    if flag_plot:
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(color_binary)
        ax2.set_title('get color binary Result()', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    return combined_binary, color_binary

# get warped image (bird view)
'''
In:
    img  numpy array   [ysize, xsize, 1] binary image(True:255, False:0)
    M    numpy array   [3,3]             transform parameter which is returned from cv2.getPerspectiveTransform 
Out:
    birds_view numpy array [ysize, xsize, 1] binary image(True:255, False:0)
'''
def get_bird_view(img, M, flag_plot=False):
    birds_view = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
    if flag_plot:
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        f.tight_layout()
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(birds_view, cmap='gray')
        ax2.set_title('get birds view', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    
    return birds_view

# get polyfit curve
'''
In:
    binary_warped     numpy array    [ysize, xsize, 1]numpy array [n]   x pixel of found as left lane     binary image(True:255, False:0)
Out:
    leftx    numpy array  [The number of found x lane pixels]   x pixel of found as left lane. 
    lefty    "
    rightx   "
    righty   "
    out_img  numpy array [ysize, xsize, 3]  Input binary image with rectangle windows. 
    histogram
'''
def find_lane_pixels_with_windows(binary_warped, flag_plot):
    # Take a histogram of the bottom half of the image
    # あくまで対象範囲全部合わせたx方向に変化するデータ
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # mask left side and right side
    thr_left = binary_warped.shape[1] // 10
    thr_right = binary_warped.shape[1] * 9 // 10
    histogram[0:thr_left] = 0
    histogram[thr_right:] = 0
    
    if flag_plot:
        plt.plot(histogram)
        plt.show()
        
    # Create an output image to draw on and visualize the resul
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])              #ヒストグラムが最大となる左側のインデックス
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint  #ヒストグラムが最大となる右側のインデックス(midpointを足すことで全体の中でのインデックスへ)

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    
    # Set the width of the windows +/- margin
    window_width = 200
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)  
    
    # Set minimum number of pixels found to recenter window
    minpix = window_height * window_width / 25
    
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero() #バイナリ画像の非0要素のインデックス
    nonzeroy = np.array(nonzero[0]) #行方向のインデックス
    nonzerox = np.array(nonzero[1]) #列歩行のインデックス
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base     #最大となったところをベースとする
    rightx_current = rightx_base   #最大となったところをベースとする

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low  = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        # currentは一段前の推定値に影響を受ける(最初は全体のヒストグラムから得る)
        win_xleft_low   = leftx_current - window_width//2  #マージンを取ることで、全体のなかであたりを付けた後の幅を決められる
        win_xleft_high  = leftx_current + window_width//2
        win_xright_low  = rightx_current - window_width//2
        win_xright_high = rightx_current + window_width//2
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0, 255, 0), 3) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 3) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        #今の対象とするyとxの範囲へ、先に調べた全体のnonzero要素から抜き出し、
        # 抜き出した後のｙのインデックスをえる
        good_left_inds = (
            (nonzeroy >= win_y_low) & 
            (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  
            (nonzerox < win_xleft_high)).nonzero()[0] 
        good_right_inds = (
            (nonzeroy >= win_y_low) & 
            (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  
            (nonzerox < win_xright_high)).nonzero()[0] #xも同じように
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        # 十分に見つけられていたら、平均値を次のleftの予想に反映させる
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print('can\'t find lane')
        pass

    # Extract left and right line pixel positions
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img, histogram

'''
In:
    binary_warped     numpy array    [ysize, xsize, 1]numpy array [n]   x pixel of found as left lane     binary image(True:255, False:0)
    left_fit    numpy array [3]                return of  np.polyfit
    right_fit             "  
Out:
    leftx    numpy array  [The number of found x lane pixels]   x pixel of found as left lane. 
    lefty    "
    rightx   "
    righty   "
    out_img  numpy array [ysize, xsize, 3]  Input binary image with rectangle windows. 
    histogram
'''
def find_lane_pixels_with_previous_polyfit(binary_warped, left_fit, right_fit, flag_plot):
    window_width = 200
    
    # Grab activated pixels
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # サーチの範囲を前回の近似曲線をつかう
    # マージンの考え方はおなじ
    # slidingwndowよりも分解のが高く、さらに不安定になりにくい(windowsで高さを1にするとたぶん不検出が増える)
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2)   + left_fit[1]*nonzeroy  + left_fit[2]  - window_width//2)) & 
                      (nonzerox < (left_fit[0]*(nonzeroy**2)   + left_fit[1]*nonzeroy  + left_fit[2]  + window_width//2)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - window_width//2)) & 
                       (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + window_width//2)))
    
    # Again, extract left and right line pixel positions
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #--------------------
    # Create an output image to draw on and visualize the resul
    #--------------------
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Draw the windows on the visualization image
    resolution_plot_y = 10 #pixel

    plot_y = np.arange(0, binary_warped.shape[0], resolution_plot_y)
    plot_x_leftlane_left   = np.round(left_fit[0]*(plot_y**2)   + left_fit[1]*plot_y   + left_fit[2]   - window_width/2).astype('int32')
    plot_x_leftlane_right  = np.round(left_fit[0]*(plot_y**2)   + left_fit[1]*plot_y   + left_fit[2]   + window_width/2).astype('int32')
    plot_x_rightlane_left  = np.round(right_fit[0]*(plot_y**2)  + right_fit[1]*plot_y  + right_fit[2]  - window_width/2).astype('int32')
    plot_x_rightlane_right = np.round(right_fit[0]*(plot_y**2)  + right_fit[1]*plot_y  + right_fit[2]  + window_width/2).astype('int32')

    for idx, y in enumerate(plot_y):
        y_low  = y
        y_high = y + resolution_plot_y
        if y_high > binary_warped.shape[0]:
            y_high = binary_warped.shape[0]

        xleftlane_left   = plot_x_leftlane_left[idx]
        xleftlane_right  = plot_x_leftlane_right[idx]
        xrightlane_left  = plot_x_rightlane_left[idx]
        xrightlane_right = plot_x_rightlane_right[idx]

        pts = np.array( [ [xleftlane_left,y_high], [xleftlane_left,y_low], [xleftlane_right, y_low], [xleftlane_right,y_high] ] )
        cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0)) 
        pts = np.array( [ [xrightlane_left,y_high], [xrightlane_left,y_low], [xrightlane_right, y_low], [xrightlane_right,y_high] ] )
        cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0)) 
    
    #--------------------
    # dummy for plot
    #--------------------
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    return leftx, lefty, rightx, righty, out_img, histogram

'''
In:
    binary_warped     numpy array    [ysize, xsize, 1]numpy array [n]   x pixel of found as left lane     binary image(True:255, False:0)
    line              class Line     1
Out:
    left_fit    numpy array [3]                return of  np.polyfit
    right_fit             "
    out_img     numpy array [ysize, xsize, 3]  Input binary image with rectangle windows, and polyfit curves are drawn. 
'''
def fit_polynomial(binary_warped, line, flag_plot):
    # Find our lane pixels first
    if line.detected == False:
        leftx, lefty, rightx, righty, out_img, histogram = find_lane_pixels_with_windows(binary_warped, flag_plot)
    else:
        leftx, lefty, rightx, righty, out_img, histogram = find_lane_pixels_with_previous_polyfit(binary_warped, line.left_fit, line.right_fit, flag_plot)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty, rightx, 2)

    #----------------
    # temp
    #----------------
    line.detected = True
    line.left_fit  = left_fit
    line.right_fit = right_fit


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    if flag_plot:
        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        plt.imshow(out_img)
        plt.show()

    return left_fit, right_fit, out_img, histogram

# get_horizontal_gap_from_lane
'''
In:
    left_fit    numpy array [3]                return of  np.polyfit
    right_fit             "
    shape        numpy array [2]               input image size 
Out:
    gap_m       1                               distance from center of lane(+:right, -:left)
'''
def get_horizontal_gap_from_lane(left_fit, right_fit, shape, flag_plot):
    
    ysize = shape[0]
    xsize = shape[1]
    
    left  = left_fit[0]*ysize**2  + left_fit[1]*ysize  + left_fit[2]
    right = right_fit[0]*ysize**2 + right_fit[1]*ysize + right_fit[2]
    
    lane_width_pixel = right - left
    gap_pixel  = (right + left)/2 - xsize/2
    gap_m      = gap_pixel * 3.7 / lane_width_pixel
    
    if flag_plot:
        print('lane_width_pixel:%.0f' % lane_width_pixel )
        print('gap_m:%.1f' % gap_m)

    return gap_m

# get curvature(in meter)
'''
In:
    left_fit    numpy array [3]                return of  np.polyfit
    right_fit             "
    y_eval      float                          y of calculate curvature 
Out:
    left_curved float       [1]                 curvature of left lane(m)
    right_curved    "
'''
def measure_curvature_real(left_fit, right_fit, y_eval, flag_plot):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    if flag_plot:
        print('curvature left:%.1fm  right:%.1fm' % (left_curverad, right_curverad))
    
    return left_curverad, right_curverad


# draw lane region
'''
In:
    warped  numpy array   [ysize, xsize, 1]  binary image(True:255, False:0)
    undist  numpy array   [ysize, xsize, 3]  
    Minv    numpy array   [3,3]             transform parameter which is returned from cv2.getPerspectiveTransform 
    left_fit    numpy array [3]                return of  np.polyfit
    right_fit             "
Out:
    image_draw_lane    numpy array   [ysize, xsize, 3]    detected lane region is drawn with green
'''
def draw(warped, undist, Minv, left_fit, right_fit, flag_plot):
    # Create an image to draw the lines on
    warp_zero  = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # get x of plyfit curve
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx  = left_fit[0]*ploty**2  + left_fit[1]*ploty  + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    image_draw_lane = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
    return image_draw_lane


'''
In:
    image_draw_lane    numpy array   [ysize, xsize, 3]  detected lane region is drawn with green
    gap_m              float                            distance from center of lane(+:right, -:left)
    left_curved        float       [1]                  curvature of left lane(m)
    right_curved        "
Out:
'''
def draw_statistics(image_draw_lane, gap_m, left_curverad, right_curverad):
    cv2.putText(image_draw_lane, ("gap from center(m):%.2f" % gap_m), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,200), 2, cv2.LINE_AA)
    cv2.putText(image_draw_lane, ("left lane curvature(m): %.0f" % left_curverad), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,200), 2, cv2.LINE_AA)
    cv2.putText(image_draw_lane, ("right lane curvature(m):%.0f" % right_curverad), (20,190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,200), 2, cv2.LINE_AA)
    return image_draw_lane
