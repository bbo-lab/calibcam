import os
from pathlib import Path

import numpy as np
import argparse
from calibcamlib.camerasystem import Camerasystem
from calibcam.detection import Detections

from bbo.geometry import RigidTransform
from bbo import label_lib
import yaml

def main(label_file,
         path = None,
         offsets = None):
    label_file = Path(label_file)

    if path is None:
        path = os.getcwd()
    path = Path(path)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        path = path / "detections.yml"

    labels = label_lib.load(label_file, v0_format=False)
    frame_idxs = label_lib.get_labeled_frame_idxs(labels)
    n_cams = label_lib.get_n_cams(labels)

    if offsets is None:
        offsets = [0] * n_cams
    offsets = np.array(offsets, dtype=int)
    offsets -= offsets[0]

    marker_names = label_lib.get_labels(labels)
    marker_ids = np.unique([int(mn.split("_")[1]) for mn in marker_names])
    time_base = np.arange(frame_idxs.max()+1)
    marker_coords, frame_idxs = label_lib.to_numpy(labels, extract_labels=marker_names,
                                                   time_bases=[time_base - o for o in offsets],
                                                   time_bases_complete=True)
    frame_idxs_full = np.array([frame_idxs + o for o in offsets])

    detection_idxs = np.arange(len(frame_idxs))

    detections = Detections.from_array({
            "marker_coords": marker_coords,
            "marker_ids": marker_ids,
            "detection_idxs": detection_idxs,
            "frame_idxs": frame_idxs_full,
        })
    detections.reset_detection_idxs()
    detections.to_file(path)

    # for i_cam in range(n_cams):
    #     for i_mn, mn in enumerate(marker_names):
    #         for i_fr, fr_idx in enumerate(used_frame_idxs):
    #             if fr_idx in labels["labels"][mn]:
    #                 corners[i_fr, i_mn] = labels["labels"][mn][fr_idx]["coords"][i_cam]
    #
    #     dpath = f"{path.as_posix()}_{i_cam:03d}.yml"
    #     with open(dpath, "w") as f:
    #         yaml.dump({
    #             "corners": corners.tolist(),
    #             "used_frames_ids": used_frame_idxs.tolist(),
    #             "rec_file_name": label_file.as_posix(),
    #         }, f, default_flow_style=True)
    #     print(f"Saved {dpath}")
    #
    #     corners[:] = np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("LABEL_FILE", type=str)
    parser.add_argument("--detection_path", type=str, default=None)
    parser.add_argument("--frame_offsets", type=int, nargs="*", default=None)

    args = parser.parse_args()

    main(args.LABEL_FILE, path=args.detection_path, offsets=args.frame_offsets)
