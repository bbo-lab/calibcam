#!/usr/bin/env python3
import argparse
import yaml
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
    parser.add_argument('--pipelines', type=str, required=False, nargs='*', default=None,
                        help="Add pipeline readable by bbo-svidreader. "
                             "The final output of the pipeline is used for calibration.")
    parser.add_argument('--config', type=str, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--board', type=str, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--model', type=str, required=False, nargs=1, default=["pinhole"], help="")
    parser.add_argument('--frame_step', type=int, required=False, nargs=1, default=[None], help="")
    parser.add_argument('--start_frame_indexes', type=int, required=False, nargs='*', default=None, help="")
    parser.add_argument('--frames_masks', type=str, required=False, nargs=1, default=[None], help="A .npy file")
    parser.add_argument('--optimize_only', required=False, default=None, action="store_true", help="")
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
        assert len(args.videos) == len(args.start_frame_indexes), "number of start_frame_indexes " \
                                                                  "does not match number of videos!"
        opts['start_frame_indexes'] = args.start_frame_indexes
    # Use frames_masks together with start_frames_indexes only after fully understanding their use!
    if args.frames_masks[0] is not None:
        opts['init_frames_masks'] = args.frames_masks[0]

    # Load details from config file
    if isinstance(args.config[0], str):
        with open(args.config[0], "r") as file:
            config_dict = yaml.safe_load(file)
            assert len(args.videos) == len(config_dict["cameras"]), "Number of cameras provided in the config file " \
                                                                    "does not match the number of videos!"
    else:
        config_dict = {'cameras':
                           [{'model': args.model[0]} for _ in args.videos]
                       }

    # Write options to file for later editing. File in data_path will be automatically included and supersedes defaults
    if isinstance(args.write_opts[0], str):
        cameras = config_dict['cameras']
        save_opts = helper.deepmerge_dicts(opts, calibrator_opts.get_default_opts([cam['model'] for cam in cameras]))
        np.save(args.write_opts[0] + "/opts.npy", save_opts, allow_pickle=True)
        print(f"Options written to {args.write_opts[0] + '/opts.npy'}")

    # Run calibration
    if isinstance(args.videos[0], str):
        recFileNames = sorted(args.videos)

        if args.pipelines is None:
            recPipelines = None
        elif len(args.pipelines) == len(recFileNames):
            recPipelines = [args.pipelines[args.videos.index(rec)] for rec in recFileNames]
        elif len(args.pipelines) == 1:
            recPipelines = args.pipelines * len(recFileNames)
        else:
            print("Sorry, the number of pipelines does not match the number of videos!")
            raise RuntimeError

        cameras = config_dict['cameras']
        opts = helper.deepmerge_dicts(opts, calibrator_opts.get_default_opts([cam['model'] for cam in cameras]))

        # 'internals' will be deleted from the opts dictionary later.
        opts['internals'] = [None for _ in cameras]
        for i_cam, cam in enumerate(cameras):
            if 'calibration_file' in cam.get('internals', {}):
                idx_2get = cam['internals']['calibration_cam_idx']
                opts['internals'][i_cam] = np.load(cam['internals']['calibration_file'],
                                                   allow_pickle=True).item()['calibs'][idx_2get]
            if not cam.get('internals', {}).get('optimize', True):
                opts['free_vars'][i_cam]['A'][:] = False
                opts['free_vars'][i_cam]['xi'] = False
                opts['free_vars'][i_cam]['k'][:] = False

        print(f"Camera models: {[cam['model'] for cam in cameras]}")
        calibrator = CamCalibrator(recFileNames, pipelines=recPipelines, board_name=args.board[0], opts=opts, data_path=args.data_path[0])
        calibrator.perform_multi_calibration()
        print("Camera calibrated")

    toc = timeit.default_timer()

    print(f"Overall procedure took {toc - tic} s")

    return


if __name__ == '__main__':
    main()
