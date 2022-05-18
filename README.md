# calibcam
A charuco based calibrator for camera setups (intrinsic and extrinsic coordinates).

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