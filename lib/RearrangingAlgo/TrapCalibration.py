"""
TrapCalibration.py

An important I/O for the rearranging device. Basically an image processing project.

Notes
-----
a. Able to detect atom location and provide mapping between spatial and freq basis.
b. Able to align AOD generated tweezer beams to actual atom location.

Contributors
------------
Chun-Wei Liu (cl3762)

"""
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import pickle


## For trap location detection

def trapLocator(path):
    """
    The main function that will
    1. Load trap images
    2. Detect traps from images
    3. Mark traps on images
    4. Save the location data as a pickle file.
    """
    ## Define image file path
    img_path_target = f"{path}_target.png"
    img_path_reservior = f"{path}_reservior.png"
    img_path_overall = f"{path}_overall.png"
    _, overall_image = __imageProcess(img_path_overall)

    ## Result image and keypoints
    target_keypoints = __detectTrapLocation(img_path_target, 'target')
    reservior_keypoints = __detectTrapLocation(img_path_reservior, 'reservior')
    overall_keypoints = __detectTrapLocation(img_path_overall, 'overall')

    ## Write trap location without AOD calibration
    ## The rescaled keypoints if needed
    __write_txt_test(overall_image, target_keypoints, path, 'target')
    __write_txt_test(overall_image, reservior_keypoints, path, 'reservior')
    __write_txt_test(overall_image, overall_keypoints, path, 'overall')

    ## Write trap location
    # __write_txt(target_keypoints, path, 'target')
    # __write_txt(reservior_keypoints, path, 'reservior')
    # __write_txt(overall_keypoints, path, 'overall')
    #print([[point.pt[0], point.pt[1]] for point in keypoints])

    return overall_keypoints, target_keypoints, reservior_keypoints

def __imageProcess(path):
    """
    Preprocess the aquired EMCCD trap image for future uses.
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_original = image.copy()

    # Make image binary
    thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Close image to get blocked patterns, or "binary image", trap color = 255
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return image_original, closing 

## For test atom image, renormalize to 75 ~ 85 MHz
def __keypointsRescale(keypoint, image_height, image_width, min_freq = 75, max_freq = 85):
    """
    Rescale the aquired trap images to 75 ~ 85 (MHz). 
    
    Note
    ----
    Here, we require all three images (target, reservior, overall) to be the same size.
    """
    return keypoint * (max_freq - min_freq) / max(image_height, image_width) + min_freq

def __detectTrapLocation(path, trap_mode):
    """
    Detecting trap locations from aquired EMCCD camera images.
    """
    """
    Detecting trap locations from aquired EMCCD camera images.
    """
    image_original, image_processed = __imageProcess(path)
    blob_params = cv2.SimpleBlobDetector_Params()

    # images are converted to many binary b/w layers. Then 255 searches for dark blobs, 0 searches for bright blobs. Or you set the filter to "false", then it finds bright and dark blobs, both.
    blob_params.filterByColor = True
    blob_params.blobColor = 255

    # Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
    blob_params.filterByArea = True
    blob_params.minArea = 1 # Highly depending on image resolution and dice size
    blob_params.maxArea = 1500. # float! Highly depending on image resolution.

    # A critical step to turn off all the filters.
    blob_params.filterByArea = False  # Can be changed to compute rel to diffraction limit
    blob_params.filterByCircularity = False
    blob_params.filterByConvexity = False
    blob_params.filterByInertia = False

    # Create a simple blob detector
    detector = cv2.SimpleBlobDetector_create(blob_params)

    # Detect blobs
    blobs = detector.detect(image_processed)

    # Get blob statistics.
    blob_count = len(blobs)
    blob_centers = np.zeros((2, blob_count))
    blob_diameters = np.zeros(blob_count)
    for (blob_idx, blob) in enumerate(blobs):
        blob_centers[:, blob_idx] = blob.pt
        blob_diameters[blob_idx] = blob.size

    print(f'Trap Type: {trap_mode}; Number of traps: {blob_count}\n')

    # Plot out identified traps
    fig, ax = plt.subplots(figsize=(10, 8), dpi = 100)
    ax.imshow(image_original, cmap='gray')

    # Assign a color to plot
    if trap_mode == "target": # green
       color = 'green'
    elif trap_mode == "reservior": # red
        color = 'red'
    else: # red
        color = 'red'

    for blob_idx in range(blob_count):
        patch = matplotlib.patches.Circle(
            (float(blob_centers[0, blob_idx]), float(blob_centers[1, blob_idx])),
            radius=float(blob_diameters[blob_idx]),
            color=color,
            linewidth=1,
            fill=None
        )
        ax.add_patch(patch)

    plt.savefig(f"{path}_{trap_mode}_result.png")
    return ax, blobs

def __write_txt_test(img, keypoints, path, trap_mode):
    """
    Write trap locations into a pickle file.
    """
    img_height, img_width, channel = img.shape
    print(f'img_height: {img_height}, {img_width}')

    position_list = [[__keypointsRescale(point.pt[0], img_height, img_width), __keypointsRescale(point.pt[1], img_height, img_width)] for point in keypoints]

    with open(f'{path}_{trap_mode}.pkl', 'wb') as f:
        pickle.dump(position_list, f)
    

def __write_txt(keypoints, path, trap_mode):
    """
    Write trap locations into a pickle file.
    """
    position_list = [[point.pt[0], point.pt[1]] for point in keypoints]

    with open(f'{path}_{trap_mode}.pkl', 'wb') as f:
        pickle.dump(position_list, f)

## For AOD calibration
## Create coordinate mapping between pixel basis (EMCCD image) and freqiency basis (AOD-AWG rf signals).

## Note
## ----
## Linear mapping should work fine since we assumed that calibration lights are deflected in small angles.
## Those two sets of coordinates are in R2.


def freq_coordinate_mapping(pixel_coor, matching):
    """
    Mapping trap location in EMCCD basis to AWG-AOD frequency basis.

    Paramaters
    ----------
    pixel_coor : np.array, list
        Trap points in EMCCD pixel coordinates.
    matching : np.array, list
        The calibration data in the form of {{pixel, freq}...}

    Returns
    -------
    freq_coor : list
        Trap points in frequency pixel coordinates.
    """
    mapping_pixel2freq_x1, mapping_pixel2freq_x2 = __create_pixel2freq_coordinate_mapping(matching)

    freq_coor = []
    for coor in pixel_coor:
        freq_coor.append([mapping_pixel2freq_x1(coor[0]), mapping_pixel2freq_x2(coor[1])])

    return freq_coor 


def __create_pixel2freq_coordinate_mapping(matching):
    """
    Mapping (interpolate) between EMCCD pixel to AWG freq coordinates.
    
    Parameters
    ----------
    pixel_coor : np.array, list
        Trap points in EMCCD pixel coordinates.
    freq_coor : np.array, list
        Trap points in frequency coordinates.
    matching : np.array, list
        The calibration data in the form of {{pixel, freq}...}

    Returns
    -------
    mapping_pixel2freq_x1 : function
        The mapping function for x1 coordinate.
    mapping_pixel2freq_x2 : function
        The mapping function for x2 coordinate.

    Note
    ----
    We seperate 2D coordinates into x1 and x2.
    """
    
    # Extract data
    matching_pixel_x1 = []
    matching_freq_x1 = []
    matching_pixel_x2 = []
    matching_freq_x2 = []
    
    for match in matching:
        # For x1 coordinates
        matching_pixel_x1.append(match[0][0])
        matching_freq_x1.append(match[1][0])
        # For x2 coordinates
        matching_pixel_x2.append(match[0][1])
        matching_freq_x2.append(match[1][1])
    
    mapping_pixel2freq_x1 = interp1d(matching_pixel_x1, matching_freq_x1, kind='cubic')
    mapping_pixel2freq_x2 = interp1d(matching_pixel_x2, matching_freq_x2, kind='cubic')

    return mapping_pixel2freq_x1, mapping_pixel2freq_x2


def trap_alignment_beam_checker():
    """
    Sweeping the alignment beam across trap locations.
    """
    print("Beam checker")
    matching = []
    return matching


def alignment_beam_tracker():
    """
    Taking the real time camera image, and detect if the alignment beam is overalapped with trap location.
    
    Note
    ----
    If the alignment beam is aligneted to trap location, then record the pixel coordinate and frequency as matching data.
    
    Source
    ------
    [1] https://github.com/ndrwnaguib/LaserPointerTracking/blob/master/track_laser.py
    [2] https://stackoverflow.com/questions/50500429/opencv-python-laser-dot-tracking-extracting-x-and-y-coordinates-and-store-it-to
    [3] https://github.com/ch-tseng/report_BackgroundSubtractor/blob/master/compare.py
    """
    ## Set up background substracter
    backSub= cv2.createBackgroundSubtractorMOG2() # Can also be KNN

    cap = cv2.VideoCapture(0)

    beam_location = []
    while (1):

        ## Take each frame
        ret, frame = cap.read()

        if frame is None:
            break
        
        ## Apply background substraction
        frame = backSub.apply(frame)

        ## The tracker algo
        ## Applying a mask via HSV color mode
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 0, 255])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)

        # Draw a circle to mark the beam
        cv2.circle(frame, maxLoc, 20, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Track Laser', frame)

        moments = cv2.moments(hsv[:, :, 2])
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
        print(f'Beam location: ({x}, {y})')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()