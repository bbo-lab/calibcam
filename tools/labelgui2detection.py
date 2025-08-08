import os
from pathlib import Path

import numpy as np
import argparse
from calibcamlib.camerasystem import Camerasystem

from bbo.geometry import RigidTransform
from bbo import label_lib
import yaml

def main(label_file, path = None):
    label_file = Path(label_file)

    if path is None:
        path = os.getcwd()
    path = Path(path)
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        path = path / f"detection"

    labels = label_lib.load(label_file, v0_format=False)
    n_cams = label_lib.get_n_cams(labels)

    marker_names = label_lib.get_labels(labels)
    used_frame_idxs = label_lib.get_labeled_frame_idxs(labels)

    corners = np.full((len(used_frame_idxs), len(marker_names), 2), np.nan)

    for i_cam in range(n_cams):
        for i_mn, mn in enumerate(marker_names):
            for i_fr, fr_idx in enumerate(used_frame_idxs):
                if fr_idx in labels["labels"][mn]:
                    corners[i_fr, i_mn] = labels["labels"][mn][fr_idx]["coords"][i_cam]

        dpath = f"{path.as_posix()}_{i_cam:03d}.yml"
        with open(dpath, "w") as f:
            yaml.dump({
                "corners": corners.tolist(),
                "used_frames_ids": used_frame_idxs.tolist(),
                "rec_file_name": label_file.as_posix(),
            }, f, default_flow_style=True)
        print(f"Saved {dpath}")

        corners[:] = np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("LABEL_FILE", type=str)
    parser.add_argument("--detection_path", type=str, default=None)

    args = parser.parse_args()

    main(args.LABEL_FILE, path=args.detection_path)
