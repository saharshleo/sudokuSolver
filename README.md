# Sudoku Solver
Hello World of Computer Vision and Machine Learning   

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Tech Stack](#tech-stack)
  * [File Structure](#file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [To Do](#to-do)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [Resources](#resources)
* [License](#license)


<!-- ABOUT THE PROJECT -->
## About The Project 
![DemoImg](https://github.com/saharshleo/sudokuSolver/blob/master/assets/demo.png)

_Solve sudoku puzzle using computer vision and machine learning. Currently(v1.0) solves sudoku from image and overlays missing digits on it_  
***
**Steps followed in this process:**
1. Preprocess image (resize, grayscale, blur, threshold, dilation/opening)  

|![original](https://github.com/saharshleo/sudokuSolver/blob/master/test_imgs/sudoku9.png)|![resized](https://github.com/saharshleo/sudokuSolver/blob/master/assets/resized_image.png)|![gray](https://github.com/saharshleo/sudokuSolver/blob/master/assets/gray.png)|![blur](https://github.com/saharshleo/sudokuSolver/blob/master/assets/blur.png)|![threshold](https://github.com/saharshleo/sudokuSolver/blob/master/assets/threshold.png)|![negate](https://github.com/saharshleo/sudokuSolver/blob/master/assets/negate.png)|![dilate](https://github.com/saharshleo/sudokuSolver/blob/master/assets/processed_img.png)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Original|Resized|GrayScale|Gaussian Blur|Adaptive Threshold|Negate|Dilation|  

2. Draw external contours.  

|![contour](https://github.com/saharshleo/sudokuSolver/blob/master/assets/external_contours.png)|
|:---:|
|External Contours|  
> Current method assumes that the sudoku has largest contour in given image  

3. Infer corners of largest contour(sudoku puzzle) and Perspective transformation based on corners

|![corner](https://github.com/saharshleo/sudokuSolver/blob/master/assets/corners.png)|![transform](https://github.com/saharshleo/sudokuSolver/blob/master/assets/warped_resized.png)|
|:---:|:---:|
|Corners|Transform|

4. Infer grid from transformed image. 

|![grid](https://github.com/saharshleo/sudokuSolver/blob/master/assets/grid_img.png)|
|:---:|
|Grid|
> Current method just divides the transformed image into number of cells in sudoku puzzle i.e 81

5. Extract digits from cell by finding largest connected pixel structure in mid part of cell. 
Scale and centre each digit, so that it becomes apt for prediction using neural network 

|![digits](https://github.com/saharshleo/sudokuSolver/blob/master/assets/extracted_digits.png)|
|:---|
|Extracted Digits|

6. Classify Digits using trained model
7. Solve the grid using backtracking algorithm
8. Draw the numbers on black background, inverse transform it and add it to original image

|![missing_digit](https://github.com/saharshleo/sudokuSolver/blob/master/assets/digit_img.png)|![solution](https://github.com/saharshleo/sudokuSolver/blob/master/assets/final_img.png)|
|:---:|:---:|
|Missing Digits|Solution|

### Tech Stack
This section should list the technologies you used for this project. Leave any add-ons/plugins for the prerequisite section. Here are a few examples.
* [OpenCV](https://opencv.org/)
* [Tensorflow-Keras](https://www.tensorflow.org/guide/keras/sequential_model)  
* [Backtracking](https://www.geeksforgeeks.org/backtracking-algorithms/)

### File Structure
    .
    ├── run.py                  # Driver code
    ├── utils                   # helper classes
    │   ├── img_processing.py   # helper functions for image processing
    │   ├── classify_digit.py   # helper functions for digit classification
    │   └── solve_sudoku.py     # helper functions to solve partially filled sudoku
    ├── test_imgs               # images for testing
    ├── assets                  # for readme
    ├── digit_classifier        # codes to train digit classifier
    ├── LICENSE
    └── README.md 
    
<!-- GETTING STARTED -->
## Getting Started

### Prerequisites 
Tested on - 
* Tensorflow v2.2.0
* OpenCV v4.1.0
* numpy v1.18.5
* scipy v1.4.1
* tabulate v0.8.7

### Installation
1. Clone the repo
```sh
git clone https://github.com/saharshleo/sudokuSolver.git
```
2. Download the pretrained model from releases [v1.0](https://github.com/saharshleo/sudokuSolver/releases/tag/v1.0)
3. Extract the model inside `sudokuSolver/models/`


<!-- USAGE EXAMPLES -->
## Usage
```
cd /path/to/sudokuSolver
```

```
python run.py
```
> For viewing the journey of image
```
python run.py --show True
```


<!-- RESULTS -->
## Results
 
![**Inference**](https://github.com/saharshleo/sudokuSolver/blob/master/assets/demo2.png)    


<!-- TO DO -->
## To Do
- [x] v1.0 Solve using Image processing and Machine learning
- [ ] v1.1 Training on own data since model trained on mnist dataset did not gave acceptable results
- [ ] v1.2 Solving on video stream
- [ ] v1.3 Robust method for infering grid
- [ ] v1.4 Different approach for extracting digits robust to lighting variations
- [ ] v1.5 Able to recognize rotated sudoku's
- [ ] v2.0 GUI game  


<!-- TROUBLESHOOTING -->
## Troubleshooting
* Changing parameters like `--process`, `--resize`, `--margin` can prove to be effective for some images and models  
> Dilation for `test_imgs/sudoku5.jpg`

|![corners-dilation](https://github.com/saharshleo/sudokuSolver/blob/master/assets/corners_dilation_sudoku5.png)|![grid-dilation](https://github.com/saharshleo/sudokuSolver/blob/master/assets/grid_img_dilation_sudoku5.png)|
|:---:|:---:|
|Corners Dilation|Infered Grid|  

> Opening for `test_imgs/sudoku5.jpg`

|![corners-opening](https://github.com/saharshleo/sudokuSolver/blob/master/assets/corners_opening_sudoku5.png)|![grid-opening](https://github.com/saharshleo/sudokuSolver/blob/master/assets/grid_img_opening_sudoku5.png)|
|:---:|:---:|
|Corners Opening|Infered Grid|  


<!-- CONTRIBUTORS -->
## Contributors
* [Saharsh Jain](https://github.com/saharshleo)


<!-- RESOURCES -->
## Resources  
* [Nesh Patel's Sudoku Solver](https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2) 
* [Backtracking for solving](https://www.geeksforgeeks.org/sudoku-backtracking-7/)
* [Floodfill](https://medium.com/@elvisdias/introduction-to-opencv-with-python-i-9fc72082f230)
* [Image Processing](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html)
* [Corners from points](https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/)
* [Conulutional network for digit classification](https://www.kaggle.com/dingli/digits-recognition-with-cnn-keras)


<!-- LICENSE -->
## License
Describe your [License](LICENSE) for your project. 

