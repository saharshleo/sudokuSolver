import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess_image(img, blur_type):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur_type == 'gaussian' or blur_type == 'Gaussian':
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        # cv2.imshow('gaussian_blur', blur)
    elif blur_type == 'median' or blur_type == 'Median':
        blur = cv2.medianBlur(gray, 5)
        # cv2.imshow('median_blur', blur)
    elif blur_type == 'bilateral' or blur_type == 'Bilateral':
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        # cv2.imshow('bilateral_blur', blur)

    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                        cv2.THRESH_BINARY, 11, 2)

    # Make edges and numbers white
    negate = cv2.bitwise_not(threshold)
    
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dtype='uint8')
    # kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(negate, kernel, iterations = 1)

    opening = cv2.morphologyEx(negate, cv2.MORPH_OPEN, kernel)

    return dilation, opening

def drawAllContours(processed_dilation):
    contours, hierarchy = cv2.findContours(processed_dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert to RGB for drawing contours
    processed = cv2.cvtColor(processed_dilation, cv2.COLOR_GRAY2RGB)

    all_contours_img = cv2.drawContours(processed.copy(), contours, -1, (0,255,0), 2)
    # cv2.imshow('all_contours', all_contours_img)

    return all_contours_img, contours

def drawExternalContours(processed_dilation):
    ext_contours, hier = cv2.findContours(processed_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to RGB for drawing contours
    processed = cv2.cvtColor(processed_dilation, cv2.COLOR_GRAY2RGB)

    external_contours_img = cv2.drawContours(processed.copy(), ext_contours, -1, (0,255,0), 2)
    # cv2.imshow('ext_contours', external_contours_img)

    return external_contours_img, ext_contours

def find_corners_of_largest_contour(contours, img):
    cnt = max(contours, key=cv2.contourArea)
    
    print("Contour Shape: {}".format(cnt.shape)) # N, 1, 2.......order_points require 2D Array
    cnt = cnt.reshape((cnt.shape[0], 2))    # N x 2  
    
    # Order --> tl, tr, br, bl
    corners = order_points_old(cnt)
    # corners = order_points_quad(cnt)

    # determine the most extreme points along the contour
    # extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
    # extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
    # extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    # extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
    # corners = np.array([extLeft, extRight, extTop, extBot])

    return corners

def order_points_old(pts):
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    
    # now, compute the difference between the points (y-x), the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    # return the ordered coordinates
    return np.array([tl, tr, br, bl], dtype="float32")

def order_points_quad(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[-1:-3:-1, :]
    
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

# get perspective transform matrix
def get_transform(pts):
    (tl, tr, br, bl) = pts
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # or the top-left and bottom-left
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)

    return M, maxWidth, maxHeight

def infer_grid(img):
    divided_grid = []
    grid_img = img.copy()
    height, width = grid_img.shape[:-1]
    grid_width, grid_height = width / 9, height / 9
    for j in range(9):
        for i in range(9):
            topLeft = (int(i*grid_width), int(j*grid_height))
            bottomRight = (int((i+1)*grid_width), int((j+1)*grid_height))
            cv2.rectangle(grid_img, topLeft, bottomRight, (0, 0, 255), 2)
            divided_grid.append((topLeft, bottomRight))

    return grid_img, divided_grid

def get_digits(warped, divided_grid, size_ocr=28, blur_type='gaussian', process='dilation'):
    digits = []
    processed_dilation, processed_opening = preprocess_image(warped, blur_type)
    
    processed_img = processed_opening if process == 'opening' else processed_dilation

    for grid in divided_grid:
        cell = processed_img[grid[0][1]:grid[1][1], grid[0][0]:grid[1][0]]
        digits.append(extract_digit_from_cell(cell, size_ocr))

    return digits

def extract_digit_from_cell(cell, output_size):
    height, width = cell.shape[:2]
    margin = int(np.mean([height, width])/3.5)
    bbox = find_largest_feature(cell.copy(), [margin, margin], [width-margin, height-margin])
    
    digit = cell[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]
    # scale and pad
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, output_size, 4)
    else:
        return np.zeros((output_size, output_size), np.uint8)

    return digit

def find_largest_feature(cell, topLeft=None, bottomRight=None):
    cell_copy = cell.copy()
    # floodfill --> https://medium.com/@elvisdias/introduction-to-opencv-with-python-i-9fc72082f230
    height, width = cell_copy.shape[:2]
    max_area = 0
    seed_point = (None, None)

    if topLeft == None:
        topLeft = [0, 0]

    if bottomRight == None:
        bottomRight = [width, height]

    for x in range(topLeft[0], bottomRight[0]):
        for y in range(topLeft[1], bottomRight[1]):
            # Only operate on light or white squares
            if cell_copy[y, x] == 255 and x < width and y < height:
                area = cv2.floodFill(cell_copy, None, (x, y), 64)   # why 64
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    # Colour everything grey (compensates for features outside of our middle scanning range)
    for x in range(width):
        for y in range(height):
            if x < width and y < height and cell_copy[y, x] == 255:
                cv2.floodFill(cell_copy, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

    # Highlight the main feature
    if all(seed_point):
        cv2.floodFill(cell_copy, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if cell_copy[y, x] == 64:
                cv2.floodFill(cell_copy, mask, (x, y), 0)

            # Find the bounding parameters
            elif cell_copy[y, x] == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]

    return np.array(bbox, dtype='float32')

def scale_and_centre(img, size, margin=0, background=0):
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))

# Driver Code
if __name__=="__main__":
    orig_img = cv2.imread('imgs/sudoku9.png')
    print("Image Shape: {}".format(orig_img.shape))
    
    orig_img = cv2.resize(orig_img, (640, 480))
    # cv2.imshow('original', orig_img)

    # GAUSSIAN BLUR
    start = time.time()
    processed_dilation, processed_opening = preprocess_image(orig_img, 'gaussian')
    # cv2.imshow('processed_dilation', processed_dilation)
    # cv2.imshow('processed_opening', processed_opening)
    
    # Choose img to proceed with - dilation / opening
    processed_img = processed_dilation

    all_contours_img, contours = drawAllContours(processed_img)
    external_contours_img, ext_contours = drawExternalContours(processed_img)

    # Do contour Approx for box

    corners_list = find_corners_of_largest_contour(ext_contours, processed_img)
    print("Corners: {}".format(corners_list.tolist()))
    for corner in corners_list:
        cv2.circle(external_contours_img, tuple(corner), 5, (0, 0, 255), -1)
    # cv2.imshow('corners', external_contours_img)

    transformation_matrix, width, height = get_transform(corners_list)
    warped_orig = cv2.warpPerspective(orig_img, transformation_matrix, (width, height))
    warped_opening = cv2.warpPerspective(processed_opening, transformation_matrix, (width, height))
    print("Warped Shape: {}".format(warped_orig.shape))
    # cv2.imshow('warped_orig', warped_orig)
    # cv2.imshow('warped_opening', warped_opening)

    grid_img, divided_grid = infer_grid(warped_orig)
    cv2.imshow('grid_img', grid_img)

    extracted_digits = get_digits(warped_orig, divided_grid, size_ocr=28, blur_type='gaussian', process='dilation')
    
    # kernel = np.ones((3, 3), np.uint8)
    # x_test = [(cv2.erode(img.copy(), kernel, iterations=1)).tolist() for img in extracted_digits]
    # print(all(x_test[0]))
    # model = tf.keras.models.load_model('classify_digit.model')
    # predictions = model.predict(x_test)
    # classified_digit = [np.argmax(p) for p in predictions]
    # print(classified_digit)

    # hmmm
    classified_digit = []
    kernel = np.ones((3, 3), np.uint8)
    model = tf.keras.models.load_model('classify_digit.model')
    # count = 0
    for img in extracted_digits:
        # test = np.array(img, np.uint8)
        test = (cv2.erode(img.copy(), kernel, iterations=1))
        # test = (cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, kernel))
        # cv2.imshow('dig + {}'.format(count), test)
        # count += 1
        tf.keras.utils.normalize(test, axis=1)
        if not np.any(test):
            classified_digit.append(-1)
            continue
        pred = model.predict([test.tolist()])
        classified_digit.append(np.argmax(pred[0]))
    # print(classified_digit)
    for row in range(9):
        for col in range(9):
            print(classified_digit[row*9+col], end=" ")
        print()

    kernel = np.ones((3,3),np.uint8)
    with_border = [cv2.copyMakeBorder(cv2.erode(img.copy(),kernel,iterations=1), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (255,255,255)) for img in extracted_digits]
    rows = []
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    extracted_digits = np.concatenate(rows)
    cv2.imshow('extracted_digits', extracted_digits)
    # print("Extracted_Digits Shape: {}".format(extracted_digits.shape))

    # print("Extra: {}".format(extracted_digits))
    # x_train = tf.keras.utils.normalize(extracted_digits, axis=1)
    # print("Extra: {}".format(x_train))


    # Predict....
    # model = tf.keras.models.load_model('classify_digit.model')
    # predictions = model.predict(extracted_digits)
    # classified_digit = []
    # for p in predictions:
    #     classified_digit.append(np.argmax(p))
    # print(classified_digit)

    # kernel = np.ones((3, 3), np.uint8)
    # rows = []
    # for grid in divided_grid:
    #     cell = warped_opening[grid[0][1]:grid[1][1], grid[0][0]:grid[1][0]]
    #     height, width = cell.shape[:2]
    #     margin = int(np.mean([height, width])/3.5)
    #     # bbox = find_largest_feature(cell.copy(), [margin, margin], [width-margin, height-margin])
    #     row = cv2.morphologyEx(cell[margin:(height-margin), margin:(width-margin)], cv2.MORPH_CLOSE, kernel)
    #     cv2.imshow('row', row)
    #     rows.append(row)

    # cv2.imshow('whatt', np.concatenate(rows))

    '''
    # digits = []
    # warped_opening = cv2.warpPerspective(processed_opening, transformation_matrix, (width, height))
    # warped_opening = cv2.bitwise_not(warped_opening, warped_opening)
    # for (tl, br) in divided_grid:
    #     digit = pytesseract.image_to_string(warped_opening[tl[1]:br[1], tl[0]:br[0]], lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    #     digits.append(digit)
    # # (tl, br) = divided_grid[-2]
    # # warped_opening = cv2.warpPerspective(processed_opening, transformation_matrix, (width, height))
    # # warped_opening = cv2.bitwise_not(warped_opening, warped_opening)
    # # cv2.imshow('digit', warped_opening[tl[1]:br[1], tl[0]:br[0]])
    # # digit = pytesseract.image_to_string(warped_opening[tl[1]:br[1], tl[0]:br[0]], lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    # # digits.append(digit)
    # print(digits)
    '''

    print("Gaussian: {} seconds".format(time.time() - start))


    # # BILATERAL BLUR
    # start = time.time()
    # processed_bilateral_dilation, processed_bilateral_opening = preprocess_image(orig_img, 'bilateral')
    # cv2.imshow('processed_bilateral_dilation', processed_bilateral_dilation)
    # print("Bilateral: {} seconds".format(time.time() - start))

    # # MEDIAN BLUR
    # start = time.time()
    # processed_median_dilation, processed_median_opening = preprocess_image(orig_img, 'median')
    # cv2.imshow('processed_median_dilation', processed_median_dilation)
    # print("Median: {} seconds".format(time.time() - start))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()