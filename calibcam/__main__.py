#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
from pathlib import Path

from calibcam import calibrator_opts, helper, yaml_helper
from calibcam.camcalibrator import CamCalibrator
import timeit


def main():
    print("Starting")
    tic = timeit.default_timer()
    # PArse command line arguments
    parser = argparse.ArgumentParser(description="Calibrate set of cameras")
    # Input videos. Could probably also be a driect input (no flag) now
    parser.add_argument('--videos', type=str, required=False, nargs='*', default=None, help="")
    # Process parameters, control if steps are performed or read from file
    parser.add_argument('--detection', type=str, required=False, nargs='*', default=False)
    parser.add_argument('--calibration_single', type=str, required=False, nargs='*', default=False)
    parser.add_argument('--calibration_multi', required=False, default=False, action='store_true')
    # Options
    parser.add_argument('--opts', type=str, required=False, nargs='*', default=[],
                        help="List of options files to include. Later files supersede earlier files, "
                             "commandline arguments supersede files")
    parser.add_argument('--board', type=str, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--model', type=str, required=False, nargs='*', default=False, help="")
    parser.add_argument('--frame_step', type=int, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--start_frame_indexes', type=int, required=False, nargs='*', default=None, help="")
    parser.add_argument('--frames_masks', type=str, required=False, nargs=1, default=[None], help="A .npy file")
    parser.add_argument('--optimize_only', required=False, default=None, action="store_true", help="")
    parser.add_argument('--numerical_jacobian', required=False, default=None, action="store_true", help="")
    # Other
    parser.add_argument('--pipelines', type=str, required=False, nargs='*', default=None,
                        help="Add pipeline readable by bbo-svidreader. "
                             "The final output of the pipeline is used for calibration.")
    parser.add_argument('--write_opts', type=str, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--data_path', type=str, required=False, nargs=1, default=[None], help="")

    args = parser.parse_args()

    n_cams = len(args.videos)

    # Build options from defaults and --opts parameters TODO: Currently not functional for yml, implement helpers!
    opts = calibrator_opts.get_default_opts(n_cams)
    for opts_file in args.opts:
        opts_file = Path(opts_file)
        if opts_file.suffix == ".yml":
            with open(opts_file, "r") as file:
                file_opts = yaml_helper.load_opts(yaml.safe_load(file))
        elif opts_file.suffix == ".npy":
            file_opts = np.load(opts_file, allow_pickle=True)[()]
        else:
            raise FileNotFoundError(f"{opts_file} is not supported")
        opts = helper.deepmerge_dicts(file_opts, opts)

    # Deal with process parameters
    if not any([args.detection, args.calibration_single, args.calibration_multi]):
        # No parameter has been set, which is interpreted as all desired
        args.detection, args.calibration_single, args.calibration_multi = (True, True, True)
    # Parameter with empty list means True
    if isinstance(args.detection, list):
        if len(args.detection) == 0:
            opts["detection"] = True
        else:
            opts["detection"] = args.detection
    elif args.detection:
        opts["detection"] = True
    if isinstance(args.calibration_single, list):
        if len(args.calibration_single) == 0:
            opts["calibration_single"] = True
        else:
            opts["calibration_single"] = args.calibration_single
    elif args.calibration_single:
        opts["calibration_single"] = True
    if isinstance(args.calibration_multi, list):
        if len(args.calibration_multi) == 0:
            opts["calibration_multi"] = True
        else:
            opts["calibration_multi"] = args.calibration_multi
    elif args.calibration_multi:
        opts["calibration_multi"] = True

    # Fill commandline options
    if args.optimize_only is not None:
        opts['optimize_only'] = args.optimize_only
    if args.numerical_jacobian is not None:
        opts['numerical_jacobian'] = args.numerical_jacobian
    if args.frame_step[0] is not None:
        opts['frame_step'] = args.frame_step[0]
    if args.frame_step[0] is not None:
        opts['frame_step'] = args.frame_step[0]
    if args.model:
        opts['model'] = args.model


    # It is necessary for the videos to be in sync to perform multi calibration. If some videos lag behind other videos,
    # start_frames_indexes should be provided to adjust for the lag.
    if args.start_frame_indexes is not None:
        assert len(args.videos) == len(args.start_frame_indexes), "number of start_frame_indexes " \
                                                                  "does not match number of videos!"
        opts['start_frame_indexes'] = args.start_frame_indexes

    # Use frames_masks together with start_frames_indexes only after fully understanding their use! TODO: EXPLAIN USE!!!
    if args.frames_masks[0] is not None:
        opts['init_frames_masks'] = args.frames_masks[0]

    # Fill defaults for opts that depend on other opts
    calibrator_opts.fill(opts)

    # Write options to file for later editing.
    if isinstance(args.write_opts[0], str):
        write_path = Path(args.write_opts[0])
        with open(write_path / "opts.yml", "w") as file:
            yaml.dump(yaml_helper.numpy_collection_to_list(opts), file)
        np.save(write_path / "opts.npy", opts, allow_pickle=True)
        print(f"Options written to {write_path / 'opts.{npy/yml}'}")

    recFileNames = args.videos

    if args.pipelines is None:
        recPipelines = None
    elif len(args.pipelines) == len(recFileNames):
        recPipelines = [args.pipelines[args.videos.index(rec)] for rec in recFileNames]
    elif len(args.pipelines) == 1:
        recPipelines = args.pipelines * len(recFileNames)
    else:
        print("Sorry, the number of pipelines does not match the number of videos!")
        raise RuntimeError

    calibrator = CamCalibrator(recFileNames, pipelines=recPipelines, board_name=args.board[0], opts=opts,
                               data_path=args.data_path[0])
    calibrator.perform_multi_calibration()
    print("Camera calibrated")
    calibrator.close_readers()

    toc = timeit.default_timer()

    print(f"Overall procedure took {toc - tic} s")

    return


if __name__ == '__main__':
    main()
