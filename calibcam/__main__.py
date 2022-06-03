#!/usr/bin/env python3
import argparse

from calibcam import calibrator_opts
from calibcam.camcalibrator import CamCalibrator
import timeit


def main():
    print("Starting")
    tic = timeit.default_timer()
    # PArse command line arguments
    parser = argparse.ArgumentParser(description="Calibrate set of cameras")
    parser.add_argument('--videos', type=str, required=True, nargs='*', default=None)
    parser.add_argument('--board', type=str, required=False, nargs=1, default=[None])
    parser.add_argument('--optimize_only', required=False, help="", action="store_true")
    parser.add_argument('--numerical_jacobian', required=False, help="", action="store_true")

    args = parser.parse_args()

    opts = calibrator_opts.get_default_opts()
    opts['optimize_only'] = args.optimize_only
    opts['numerical_jacobian'] = args.numerical_jacobian
    recFileNames = sorted(args.videos)
    calibrator = CamCalibrator(recFileNames, board_name=args.board[0], opts=opts)
    calibrator.perform_multi_calibration()

    toc = timeit.default_timer()

    print(f"Overall procedure took {toc-tic} s")

    return


if __name__ == '__main__':
    main()
