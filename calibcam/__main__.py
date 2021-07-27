#!/usr/bin/env python3
import sys
import argparse

from PyQt5.QtWidgets import QApplication

from .gui import MainWindow
from .board import get_board_params

def main():  
    # PArse command line arguments
    parser = argparse.ArgumentParser(description="Calibrate set of cameras")
    parser.add_argument('--videos', type=str, required=False, nargs=1) #TODO: Implement that this takes glob arg and omits GUI start
    parser.add_argument('--board', type=str, required=False, nargs=1, default='large_dual_led')
    args = parser.parse_args()
    print(args)
    
    app = QApplication(sys.argv)
    window = MainWindow(board_params=get_board_params(args.board[0]))
    sys.exit(app.exec_())
    return


if __name__ == '__main__':
    main()
