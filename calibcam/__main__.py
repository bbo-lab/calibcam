#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
from pathlib import Path

from calibcam import calibrator_opts, helper, yaml_helper
from calibcam.camcalibrator import CamCalibrator
from calibcamlib import Camerasystem
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
    parser.add_argument('--stop_frame_indexes', type=int, required=False, nargs='*', default=None, help="")
    parser.add_argument('--frames_masks', type=str, required=False, nargs=1, default=[None],
                        help="A .npy file. The frames_masks start from the start_frame_indexes and end at "
                             "the stop_frame_indexes if they are provided.")
    parser.add_argument('--optimize_only', required=False, default=None, action="store_true", help="")
    parser.add_argument('--numerical_jacobian', required=False, default=None, action="store_true", help="")
    # Other
    parser.add_argument('--pipelines', type=str, required=False, nargs='*', default=None,
                        help="Add pipeline readable by bbo-svidreader. "
                             "The final output of the pipeline is used for calibration.")
    parser.add_argument('--write_opts', type=str, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--data_path', type=str, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--gamma_correction', required=False, default=False, action="store_true", help="")
    parser.add_argument('--init_extrinsics', type=str, required=False, nargs=1, default=[None], help="")

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
    for param in ["detection", "calibration_single", "calibration_multi"]:
        if isinstance(getattr(args, param), list):
            if len(getattr(args, param)) == 0:
                opts[param] = True
            else:
                opts[param] = getattr(args, param)
        elif getattr(args, param):
            opts[param] = True

    # Fill commandline options
    if args.optimize_only is not None:
        opts['optimize_only'] = args.optimize_only
    if args.numerical_jacobian is not None:
        opts['numerical_jacobian'] = args.numerical_jacobian
    if args.model:
        opts['models'] = args.model
    if args.gamma_correction:
        opts['gamma_correction'] = True
    if args.init_extrinsics[0] is not None:
        init_extrinsics = Camerasystem.load_dict(args.init_extrinsics[0])
        opts['init_extrinsics'] = {
            'rvecs_cam': np.array([c["rvec_cam"] for c in init_extrinsics['calibs']]),
            'tvecs_cam': np.array([c["tvec_cam"] for c in init_extrinsics['calibs']])
        }

    # It is necessary for the videos to be in sync to perform multi calibration. If some videos lag behind other videos,
    # start_frames_indexes should be provided to adjust for the lag.
    if args.start_frame_indexes is not None:
        assert len(args.start_frame_indexes) == n_cams, "number of start_frame_indexes " \
                                                                  "does not match number of videos!"
        opts['start_frame_indexes'] = np.array(args.start_frame_indexes)

    if args.frame_step[0] is not None:
        opts['frame_step'] = args.frame_step[0]

    # Sometimes, it is better to use only certain portion of the video for calibration.
    # start_frame_indexes and stop_frame_indexes can be used to specify the frames to be used for calibration.
    if args.stop_frame_indexes is not None:
        assert len(args.stop_frame_indexes) == n_cams, "number of stop_frame_indexes " \
                                                                 "does not match number of videos!"
        opts['stop_frame_indexes'] = np.array(args.stop_frame_indexes)

    # Use frames_masks together with start_frames_indexes to provide the frames to be used for calibration.
    # TODO: EXPLAIN USE!!!
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
        recPipelines = args.pipelines
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
