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
    parser.add_argument('--model', type=str, required=False, nargs=1, default=["pinhole"], help="")
    parser.add_argument('--frame_step', type=int, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--start_frame_indexes', type=int, required=False, nargs='*', default=None, help="")
    parser.add_argument('--frames_masks', type=str, required=False, nargs=1, default=[None], help="A .npy file")
    parser.add_argument('--optimize_only', required=False, default=None, action="store_true", help="")
    parser.add_argument('--internals', required=False, default=[None], type=str, nargs=1, help="")
    parser.add_argument('--numerical_jacobian', required=False, default=None, action="store_true", help="")
    parser.add_argument('--write_opts', type=str, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--data_path', type=str, required=False, nargs=1, default=[None], help="")

    args = parser.parse_args()

    # Fill options. These options supersede everything (defaults, saved file)
    opts = {}
    if args.optimize_only is not None:
        opts['optimize_only'] = args.optimize_only
    if args.numerical_jacobian is not None:
        opts['numerical_jacobian'] = args.numerical_jacobian
    if args.frame_step[0] is not None:
        opts['frame_step'] = args.frame_step[0]

    # It is necessary for the videos to be in sync to perform multicalibration. If some videos lag behind other videos,
    # start_frames_indexes should be provided to adjust for the lag.
    if args.start_frame_indexes is not None:
        assert len(args.videos) == len(args.start_frame_indexes), "number of start_frame_indexes does not match number of videos"
        opts['start_frame_indexes'] = args.start_frame_indexes
    # Use frames_masks together with start_frames_indexes only after fully understanding their use!
    if args.frames_masks[0] is not None:
        opts['init_frames_masks'] = args.frames_masks[0]

    # Write options to file for later editing. File in data_path will be automatically included and supersedes defaults
    if isinstance(args.write_opts[0], str):
        save_opts = helper.deepmerge_dicts(opts, calibrator_opts.get_default_opts(args.model[0]))
        np.save(args.write_opts[0] + "/opts.npy", save_opts, allow_pickle=True)
        print(f"Options written to {args.write_opts[0] + '/opts.npy'}")

    # Run calibration
    if isinstance(args.videos[0], str):
        recFileNames = sorted(args.videos)
        opts = helper.deepmerge_dicts(opts, calibrator_opts.get_default_opts(args.model[0]))

        if args.internals[0] is not None:
            internals = np.load(args.internals[0], allow_pickle=True)[()]
            opts['internals'] = internals["calibs"]
            opts['free_vars']['A'][:] = False
            opts['free_vars']['xi'] = False
            opts['free_vars']['k'][:] = 0
            
        print(f"Camera model: {args.model[0]}")
        calibrator = CamCalibrator(recFileNames, board_name=args.board[0], opts=opts, data_path=args.data_path[0])
        calibrator.perform_multi_calibration()
        print("Camera calibrated")

    toc = timeit.default_timer()

    print(f"Overall procedure took {toc - tic} s")

    return


if __name__ == '__main__':
    main()
