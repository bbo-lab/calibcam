# calibcam
A charuco based calibrator for camera setups (intrinsic and extrinsic parameters), including omnidirectional cameras.

First, [OpenCV is used for single camera calibration.](https://opencv.org/), followed by an initial estimation of camera positions and orientations.
Subsequently, all intrinsic and extrinsic parameters are optimised for reprojection error using [Jax](https://github.com/google/jax) autograd.

See [calibcamlib](https://github.com/bbo-lab/calibcamlib) for a library for triangualtion, reprojection etc.

# Installation

## Windows

1. (If not already done:) [Install Anaconda](https://docs.anaconda.com/anaconda/install/windows/)
2. Create conda env `conda env create -f https://raw.githubusercontent.com/bbo-lab/calibcam/main/environment.yml`
3. Switch to calibcam environment: `conda activate calibcam`


# Usage

## Windows

1. From `[repository]/boards`, copy the appropriate board into the calibration video directory and rename to `board.npy`
2. Open Anaconda prompt via Start Menu
3. Switch to calibcam environment: `conda activate calibcam`
4. Run the program with `python -m calibcam --videos [LIST OF VIDEOS TO INCLUDE]`

BBO internal MATLAB use only:

Use MATLAB function `mcl = cameralib.helper.mcl_from_calibcam([PATH TO MAT FILE OUTPUT OF CALIBRATION])` from bboanlysis_m to generate an MCL file.

# Format
## Result
`multicam_calibration.npy/mat` holds a dictionary/struct with the calibration result. The filed `"calibs"` holds an array of calibration dictionarys/structs with entries
```
* 'rvec_cam': (3,) - Rotation vector of the respective cam (world->cam)
* 'tvec_cam': (3,) - Translation vector of the respective cam (world->cam)
* 'A': (3,3) - Camera matrix
* 'k': (5,) - Camera distortion coefficients
```
For further structure, refer to `camcalibrator.build_result()`

## Board
Besides the videos, each calibration folder (folder of first video) needs to contain a file `board.npy`. For the boards at BBO, files are available in the boards directory of the repository. Else, files must be created, containing a dict with the following entries:
```
* boardWidth: int - number of checkerboard squares
* boardHeight: int - number of checkerboard squares
* square_size_real: float - Absolute edge length of checkerboard squares, unit determines unit of calibration
* marker_size: float - Relative marker size
* dictionary_type: int - Aruco dictionary type
```
These values are used to create the board in the following way:
```python
board = cv2.aruco.CharucoBoard_create(board_params['boardWidth'],
                                          board_params['boardHeight'],
                                          board_params['square_size_real'],
                                          board_params['marker_size'] * board_params['square_size_real'],
                                          cv2.aruco.getPredefinedDictionary( 
                                              board_params['dictionary_type']))
```
