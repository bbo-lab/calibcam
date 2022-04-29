#!/usr/bin/env python3
import argparse
from .calibrator import Calibrator


def main():
    # PArse command line arguments
    parser = argparse.ArgumentParser(description="Calibrate set of cameras")
    parser.add_argument('--videos', type=str, required=False, nargs='*', default=None)
    parser.add_argument('--board', type=str, required=False, nargs=1, default=[None])
    args = parser.parse_args()

    calibrator = Calibrator(board_name=args.board[0])

    if args.videos is None:
        print('Starting viewer')
        from .gui import main as gui_main
        gui_main(calibrator)
    else:
        recFileNames = sorted(args.videos)
        calibrator.set_recordings(recFileNames)
        if calibrator.recordingIsLoaded:
            calibrator.perform_multi_calibration()

    return


if __name__ == '__main__':
    main()
