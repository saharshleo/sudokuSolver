import cv2
import numpy as np
import tensorflow as tf
import argparse
import time
import pprint
from tabulate import tabulate
from copy import deepcopy

from utils.img_processing import Extract_Digits
from utils.classify_digit import Classify_Digit
from utils.solve_sudoku import Solve_Sudoku

# =======IGNORE WARNINGS============================
import warnings
warnings.filterwarnings('ignore')

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ==================================================

def arg_parse():
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser(description='Sudoku Solver Module')
    parser.add_argument("--image", dest = "image", help = "path to image", default = "test_imgs/sudoku9.png", type = str)
    parser.add_argument("--model", dest = "model", help = "path to model", default = "models/loss_0298.model", type = str)
    parser.add_argument("--process", dest = "process", help = "Dilation or Opening for grid extraction", default = "dilation")
    parser.add_argument("--resize", dest = "resize", help = "(width, height) to resize to", default = (450, 450), type = tuple)
    parser.add_argument("--size", dest = "size", help = "input size to ocr model", default = 28, type = int)
    parser.add_argument("--margin", dest = "margin", help = "margin for extracted digits", default = 10, type = int)
    parser.add_argument("--grid_size", dest = "sudoku_size", help = "grid size of sudoku default is 9", default = 9, type = int)
    parser.add_argument("--show", dest = "show", help = "bool whether to show journey of image", default = False, type = bool)

    return parser.parse_args()


# Driver Code
if __name__ == "__main__":
    
    # Initialize command line arguments
    args = arg_parse()
    orig_img = cv2.imread(args.image)
    model_path = args.model
    process = args.process
    resize = args.resize
    size_ocr = args.size
    margin = args.margin
    sudoku_size = args.sudoku_size
    show_journey = args.show

    # Start
    start_time = time.time()

    img_obj = Extract_Digits(resize, size_ocr, show_journey)
    
    # Preprocess :--: resize --> gray --> blur --> threshold --> negate --> dilation/opening
    processed_img = img_obj.preprocess_image(orig_img, process=process)
    preprocessing_time = time.time()

    # External contours :--: assumption that biggest contour correspond to sudoku puzzle
    ext_contours = img_obj.draw_external_contours(processed_img)
    draw_contour_time = time.time()
    
    # Infer corners from points in largest contour
    corners = img_obj.find_corners_of_largest_contour(ext_contours, img_obj.resized_img)
    find_corner_time = time.time()

    # Perspective transformation based on corners
    transformation_matrix, width, height = img_obj.get_transform(corners)
    transformation_time = time.time()
    
    # Divide warped image into 9*9 cells :--: Need some other logic for this step
    divided_grid = img_obj.infer_grid(img_obj.warped_resized)
    grid_inference_time = time.time()
    
    # Apply floodfill in mid portion of cell to extract only digit 
    extracted_digits = img_obj.get_digits(img_obj.warped_processed, divided_grid, margin=margin)
    digit_extraction_time = time.time()

    classifier_obj = Classify_Digit(model_path, show_journey=show_journey)

    # Classify digit present in cell
    classifier_obj.classify(extracted_digits)
    pprint.pprint(classifier_obj.classified_digits)
    digit_classification_time = time.time()

    solver_obj = Solve_Sudoku(sudoku_size)

    # Solve sudoku using backtracking
    solved_sudoku = solver_obj.solve_sudoku(deepcopy(classifier_obj.classified_digits))
    solving_time = time.time()

    if(solved_sudoku):
        solver_obj.print_sudoku(solved_sudoku)
        
        # Show solution on original image
        final_img = img_obj.draw_with_solution(orig_img.copy(), solved_sudoku, classifier_obj.classified_digits, divided_grid, transformation_matrix)
        cv2.imshow('original', orig_img)
        cv2.imshow('solved', final_img)

    else:
        print("No Solution exists!!!")

    end_time = time.time()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # when show_journey is False, then only its true inference time
    # because of cv2.waitKey
    if not show_journey:
        print("\nSUMMARY:")
        table = [
            ["Preprocessing", preprocessing_time - start_time],
            ["External Contours", draw_contour_time - preprocessing_time],
            ["Finding Corners from Contour", find_corner_time - draw_contour_time],
            ["Perspecive Transformation", transformation_time - find_corner_time],
            ["Grid Inference", grid_inference_time - transformation_time],
            ["Digit Extraction", digit_extraction_time - grid_inference_time],
            ["Digit Classification", digit_classification_time - digit_extraction_time],
            ["Solve Sudoku", solving_time - digit_classification_time],
            ["Drawing on Image (if solved)", end_time - solving_time],
            ["Inference Time", end_time - start_time]
        ]
        print(tabulate(table, headers=["TASK", "TIME TAKEN (in seconds)"], tablefmt="grid"))