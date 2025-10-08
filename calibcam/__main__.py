#!/usr/bin/env python3

import argparse
import timeit
from pathlib import Path

import numpy as np
import yaml

from calibcam import calibrator_opts, helper, yaml_helper, __version__
from calibcam.camcalibrator import CamCalibrator
from calibcamlib import Camerasystem, Board


def main():
    print(f"Starting ({__version__})")
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
    parser.add_argument('--board', type=str, required=False, nargs='*', default=[None], help="")
    parser.add_argument('--model', type=str, required=False, nargs='*', default=False, help="")

    parser.add_argument('--frames_start', type=int, required=False, default=None, help="")
    parser.add_argument('--frames_end', type=int, required=False, default=None, help="")
    parser.add_argument('--frames_step', type=int, required=False, default=None, help="")
    parser.add_argument('--frames_offsets', type=int, required=False, nargs='*', default=None, help="")

    parser.add_argument('--init_extrinsics_frames', type=int, nargs='*', required=False, default=[], help="")
    parser.add_argument('--max_allowed_res', type=int, required=False, default=None, help="")

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

    opts = build_options(args, n_cams)

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

    board_params = make_board_params(args.board, recFileNames)

    calibrator = CamCalibrator(recFileNames, pipelines=recPipelines, board_params=board_params, opts=opts,
                               data_path=args.data_path[0])
    calibrator.perform_multi_calibration()
    print("Camera calibrated")
    calibrator.close_readers()

    toc = timeit.default_timer()

    print(f"Overall procedure took {toc - tic} s")

    return


def build_args_into_opts(opts, args, n_cams):
    opts = define_process_stages(args, opts)

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

    if args.frames_start is not None:
        opts['frames_start'] = args.frames_start
    if args.frames_end is not None:
        opts['frames_end'] = args.frames_end
    if args.frames_step is not None:
        opts['frames_step'] = args.frames_step

    if len(args.init_extrinsics_frames)>0:
        opts['init_extrinsics_frames'] = args.init_extrinsics_frames
    if args.max_allowed_res is not None:
        opts['max_allowed_res'] = args.max_allowed_res

    if args.frames_offsets is not None:
        assert len(args.frames_offsets) == n_cams, "Number of frames_offsets does not match number of videos!"
        opts['frames_offsets'] = np.array(args.frames_offsets)

    # Fill defaults for opts that depend on other opts
    calibrator_opts.fill(opts)

    return opts


def define_process_stages(args, opts):
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

    return opts


def build_options(args, n_cams):
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

    opts = build_args_into_opts(opts, args, n_cams)

    # Write options to file for later editing.
    if isinstance(args.write_opts[0], str):
        write_path = Path(args.write_opts[0])
        with open(write_path / "opts.yml", "w") as file:
            yaml.dump(yaml_helper.numpy_collection_to_list(opts), file)
        np.save(write_path / "opts.npy", opts, allow_pickle=True)
        print(f"Options written to {write_path / 'opts.{npy/yml}'}")

    return opts


def make_board_params(board_args, videos):
    n_videos = len(videos)

    if len(board_args) == 1:
        if board_args[0] is None:
            return None

        boards = Board.from_file(board_args[0])
        if isinstance(boards, Board):
            return [boards.get_board_params()] * n_videos
        else:
            assert len(boards) == n_videos, (f"Single board file must either contain a single board, or a list of"
                                                   f" boards of the size of the number of videos ({n_videos})")
            return [brd.get_board_params() for brd in boards]

    current_board_file = None
    board_params = []
    for board_arg in board_args:
        if board_arg.isdigit():
            assert current_board_file is not None, "Must specifiy board file before board idx!"
            board_params.append(
                Board.from_file(current_board_file, int(board_arg)).get_board_params()
            )
        else:
            current_board_file = board_arg

    assert len(board_params) == n_videos, f"Board specifications do not match number of videos ({n_videos})!"
    return board_params


if __name__ == '__main__':
    main()
