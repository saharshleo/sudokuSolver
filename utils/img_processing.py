import cv2
import numpy as np
from scipy.spatial import distance as dist

class Extract_Digits:
    def __init__(self, resize=(450, 450), size_ocr=28, show_journey=False, kernel_size=3):
        assert(kernel_size%2 != 0)
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.resize = resize
        self.size_ocr = size_ocr
        self.show_journey = show_journey
        self.resized_img = None
        self.processed_img = None
        self.warped_processed = None
        self.warped_resized = None

    def preprocess_image(self, image, process='dilation'):
        ''' return processed image '''
        self.resized_img = cv2.resize(image, self.resize)

        gray = cv2.cvtColor(self.resized_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                        cv2.THRESH_BINARY, 11, 2)
        
        negate = cv2.bitwise_not(threshold)
        if process == 'dilation':
            self.processed_img = cv2.dilate(negate, self.kernel, iterations = 1)
        elif process == 'opening':
            self.processed_img = cv2.morphologyEx(negate, cv2.MORPH_OPEN, self.kernel)
        
        if self.show_journey:
            cv2.imshow('resized_image', self.resized_img)
            cv2.imshow('gray', gray)
            cv2.imshow('blur', blur)
            cv2.imshow('threshold',threshold)
            cv2.imshow('negate', negate)
            cv2.imshow('processed_img', self.processed_img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return self.processed_img

    def draw_external_contours(self, image):
        ext_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.show_journey:
            # Convert to BGR for drawing contours
            processed = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            external_contours_img = cv2.drawContours(processed, ext_contours, -1, (0,255,0), 2)
        
            cv2.imshow('external_contours', external_contours_img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ext_contours

    def find_corners_of_largest_contour(self, contours, image):
        largest_contour = max(contours, key=cv2.contourArea)
        # print("Contour Shape: {}".format(largest_contour.shape)) # N, 1, 2
        largest_contour = largest_contour.reshape((largest_contour.shape[0], 2))    # N x 2  
        
        # Order --> topLeft, topRight, bottomRight, bottomLeft
        corners = self.order_points_old(largest_contour)
        # print("Corners: {}".format(corners.tolist()))

        if self.show_journey:
            copy = image.copy()
            for corner in corners:
                cv2.circle(copy, tuple(corner), 5, (0, 0, 255), -1)
            
            cv2.imshow('corners', copy)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return corners

    def order_points_old(self, pts):
        ''' 
        from list of points return ordered corners (tl, tr, br, bl)
        Doesn't work when sum/difference is same for corners
        '''
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

    def get_transform(self, pts):
        (tl, tr, br, bl) = pts
        
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = dist.euclidean(br, bl)
        widthB = dist.euclidean(tr, tl)
        maxWidth = max(int(widthA), int(widthB))
        
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # or the top-left and bottom-left
        heightA = dist.euclidean(tr, br)
        heightB = dist.euclidean(bl, tl)
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

        self.warped_processed = cv2.warpPerspective(self.processed_img, M, (maxWidth, maxHeight))
        self.warped_resized = cv2.warpPerspective(self.resized_img, M, (maxWidth, maxHeight))
    
        if self.show_journey:
            cv2.imshow('warped_processed', self.warped_processed)
            cv2.imshow('warped_resized', self.warped_resized)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return M, maxWidth, maxHeight

    def infer_grid(self, image):
        divided_grid = []
        grid_img = image.copy()
        height, width = grid_img.shape[:-1]
        grid_width, grid_height = width / 9, height / 9
        for j in range(9):
            for i in range(9):
                topLeft = (int(i*grid_width), int(j*grid_height))
                bottomRight = (int((i+1)*grid_width), int((j+1)*grid_height))
                cv2.rectangle(grid_img, topLeft, bottomRight, (0, 0, 255), 2)
                divided_grid.append((topLeft, bottomRight))

        if self.show_journey:
            cv2.imshow('grid', grid_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return divided_grid

    def get_digits(self, warped, divided_grid, margin=10):
        digits = []

        for grid in divided_grid:
            cell = warped[grid[0][1]:grid[1][1], grid[0][0]:grid[1][0]]
            digits.append(self.extract_digit_from_cell(cell, self.size_ocr, margin))

        return digits

    def extract_digit_from_cell(self, cell, output_size, margin):
        height, width = cell.shape[:2]
        margin = int(np.mean([height, width])/4.5)
        bbox = self.find_largest_feature(cell, [margin, margin], [width-margin, height-margin])
        
        digit = cell[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]
        # scale and pad
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]

        # Ignore any small bounding boxes
        if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
            return self.scale_and_centre(digit, output_size, margin)
        else:
            return np.zeros((output_size, output_size), np.uint8)

        return digit

    def find_largest_feature(self, cell, topLeft=None, bottomRight=None):
        cell_copy = cell.copy()

        height, width = cell_copy.shape[:2]
        max_area = 0
        seed_point = (None, None)

        if topLeft == None:
            topLeft = [0, 0]

        if bottomRight == None:
            bottomRight = [width, height]

        for x in range(topLeft[0], bottomRight[0]):
            for y in range(topLeft[1], bottomRight[1]):
                # Get largest white contour while changing all white pixels to gray
                if cell_copy[y, x] == 255 and x < width and y < height:
                    area = cv2.floodFill(cell_copy, None, (x, y), 64)
                    if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                        max_area = area[0]
                        seed_point = (x, y)

        # Colour everything grey (compensates for features outside of our middle scanning range)
        for x in range(width):
            for y in range(height):
                if x < width and y < height and cell_copy[y, x] == 255:
                    cv2.floodFill(cell_copy, None, (x, y), 64)

        mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

        # Draw the main feature
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

    def scale_and_centre(self, image, size, margin=10, background=0):
        h, w = image.shape[:2]

        def centre_pad(length):
            ''' Handles centering for a given length that may be odd or even '''
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

        image = cv2.resize(image, (w, h))
        image = cv2.copyMakeBorder(image, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
        
        return cv2.resize(image, (size, size))


    def draw_with_solution(self, orig_img, solved_sudoku, unsolved_sudoku, divided_grid, transformation_matrix):
        digit_img = np.zeros((self.warped_resized.shape[1], self.warped_resized.shape[0], 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        color = (0, 255, 0)
        font_scale = 1
        thickness = 2

        offset_x = (divided_grid[0][1][0] - divided_grid[0][0][0])//3
        offset_y = (divided_grid[0][1][1] - divided_grid[0][0][1])//3
        
        for i in range(len(solved_sudoku)):
            for j in range(len(solved_sudoku)):
                if unsolved_sudoku[i][j] == 0 and solved_sudoku[i][j] != 0:
                    label = str(solved_sudoku[i][j])
                    digit_img = cv2.putText(digit_img, label, (divided_grid[i*9+j][0][0]+offset_x, divided_grid[i*9+j][1][1]-offset_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        final_img = np.zeros((self.resize[1], self.resize[0]), np.uint8)
        final_img = cv2.warpPerspective(digit_img, transformation_matrix, (self.resize[1], self.resize[0]), final_img, cv2.WARP_INVERSE_MAP)
        final_img = cv2.add(final_img, self.resized_img)
        final_img = cv2.resize(final_img, (orig_img.shape[1], orig_img.shape[0]))

        if self.show_journey:
            cv2.imshow('digit_image', digit_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return final_img