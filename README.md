# calibcam
A charuco based calibrator for camera setups (intrinsic and extrinsic parameters), including omnidirectional cameras.

First, [OpenCV is used for single camera calibration.](https://opencv.org/), followed by an initial estimation of camera positions and orientations.
Subsequently, all intrinsic and extrinsic parameters are optimised for reprojection error using [Jax](https://github.com/google/jax) autograd.

See [calibcamlib](https://github.com/bbo-lab/calibcamlib) for a library for triangualtion, reprojection etc.

## Board
calibcam uses Charuco boards for calibration. The board needs to be presented in different angles, positions and distances to each camera. Relative camera positions are estimated from frames in which the board is visible in multiple cameras. We recommend recording with 2 fps while moving the board around, spending around a minute on each camera.

### Creation

See board.py in example folder for the generation of both printable PNG and board configuration file for calibcam.

## Installation

Install bbo-calibcam via pip, or create conda environment from environment.yml.


## Usage

1. Collect data as described in Board section. 
2. Run calibcam with `python -m calibcam --videos [LIST OF VIDEOS TO INCLUDE] --board [PATH TO BOARD.NPY file]`. We recommend keeping a copy of the board file with the videos for documentation purposes. 
3. Check number of detections per camera in the output. Values should range between 80 and 300. If too few detections are made, check recording conditions (lighting, blur ...) and collect new calibration data. If too many frames are detected, convergence may be slow or run out of memory. Reduce detections adding a frame skip with `--frame_step`.
4. Check reprojection error at the end of the output. Median errors should be <0.5px.

### BBO internal MATLAB use only:
Use MATLAB function `mcl = cameralib.helper.mcl_from_calibcam([PATH TO MAT FILE OUTPUT OF CALIBRATION])` from bboanlysis_m to generate an MCL file.

# Format
## Result
`multicam_calibration.npy/mat/yml` holds a dictionary/struct with the calibration result. The filed `"calibs"` holds an array of calibration dictionarys/structs with entries
```
* 'rvec_cam': (3,) - Rotation vector of the respective cam (world->cam)
* 'tvec_cam': (3,) - Translation vector of the respective cam (world->cam)
* 'A': (3,3) - Camera matrix
* 'k': (5,) - Camera distortion coefficients
```
For further structure, refer to `camcalibrator.build_result()`

