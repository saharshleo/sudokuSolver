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

Solve sudoku puzzle using computer vision and machine learning. Currently(v1.0) solves sudoku from image and overlays missing digits on it.  
**Steps followed in this process:**
1. Preprocess image (resize, grayscale, blur, threshold, dilation/opening)
2. Draw external contours. 
> Current method assumes that the sudoku has largest contour in given image  
3. Infer corners of largest contour(sudoku puzzle)  
4. Perspective transformation based on corners  
5. Infer grid from transformed image. 
> Current method just divides the transformed image into number of cells in sudoku puzzle i.e 81
6. Extract digits from cell by finding largest connected pixel structure in mid part of cell. 
Scale and centre each digit, so that it becomes apt for prediction using neural network  

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
    ├── digit_classifier        # codes to train digit classifier
    ├── LICENSE
    └── README.md 
    
<!-- GETTING STARTED -->
## Getting Started

### Prerequisites (Code tested on)

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
python run.py --show
```


<!-- RESULTS -->
## Results
 
[**result screenshots**](https://result.png)  
![**result gif or video**](https://result.gif)  


<!-- TO DO -->
## To Do
- [x] Task 1
- [x] Task 2
- [ ] Task 3
- [ ] Task 4


<!-- TROUBLESHOOTING -->
## Troubleshooting
* Common errors while configuring the project


<!-- CONTRIBUTORS -->
## Contributors
* [Saharsh Jain](https://github.com/saharshleo)


<!-- RESOURCES -->
## Resources  
* Refered [this](https://link) for achieving this  


<!-- LICENSE -->
## License
Describe your [License](LICENSE) for your project. 

