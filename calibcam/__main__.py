#!/usr/bin/env python3
import argparse
import numpy as np

from calibcam import calibrator_opts, helper
from calibcam.camcalibrator import CamCalibrator
import timeit


def main():
    print("Starting")
    tic = timeit.default_timer()
    # PArse command line arguments
    parser = argparse.ArgumentParser(description="Calibrate set of cameras")
    parser.add_argument('--videos', type=str, required=False, nargs='*', default=None, help="")
    parser.add_argument('--board', type=str, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--frame_skip', type=int, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--optimize_only', required=False, default=None, action="store_true", help="")
    parser.add_argument('--numerical_jacobian', required=False, default=None, action="store_true", help="")
    parser.add_argument('--write_opts', type=str, required=False, nargs=1, default=[None], help="")

    args = parser.parse_args()

    # Fill options. These options supersede everything (defaults, saved file)
    opts = {}
    if args.optimize_only is not None:
        opts['optimize_only'] = args.optimize_only
    if args.numerical_jacobian is not None:
        opts['numerical_jacobian'] = args.numerical_jacobian
    if args.frame_skip[0] is not None:
        opts['frame_skip'] = args.frame_skip[0]

    # Write options to file for later editing. File in data_path will be automatically included and supersedes defaults
    if isinstance(args.write_opts[0], str):
        save_opts = helper.deepmerge_dicts(opts, calibrator_opts.get_default_opts())
        np.save(args.write_opts[0] + "/opts.npy", save_opts, allow_pickle=True)
        print(f"Options written to {args.write_opts[0] + '/opts.npy'}")

    # Run calibration
    if isinstance(args.videos[0], str):
        recFileNames = sorted(args.videos)
        calibrator = CamCalibrator(recFileNames, board_name=args.board[0], opts=opts)
        calibrator.perform_multi_calibration()
        print("Camera calibrated")

    toc = timeit.default_timer()

    print(f"Overall procedure took {toc - tic} s")

    return


if __name__ == '__main__':
    main()
