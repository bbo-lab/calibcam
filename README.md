# calibcam
Cameracalibration derived from Arne's calibration code

# Installation

## Windows

1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/windows/)
2. Clone git@github.com:bbo-lab/calibcam.git 
3. Open Anaconda prompt via Start Menu
4. Using `cd` and `dir`, navigate to the calibcam folder INSIDE the repository (which may also be named calibcam)
5. Create conda environment using `conda env create -f environment.yml`
6. Switch to multitrackpy environment: `conda activate calibcam`
7. Add calibcam module to conda environment: `conda develop [path to your repository, including repository folder]`

You can now run the program with `python -m calibcam`
