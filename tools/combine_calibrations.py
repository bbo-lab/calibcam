from pathlib import Path
import yaml
import argparse
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R

from bbo.geometry import RigidTransform
from calibcamlib import Camerasystem
from calibcam.camcalibrator import save_multicalibration

def main():
    parser = argparse.ArgumentParser(description="Joins two calibration that where performed on the SAME set of frames"
                                                 " with COMPATIBLE BOARDS (i.e. common poses)")
    parser.add_argument('--multicamcalibs', nargs='+', type=str, required=True)
    parser.add_argument('--poses', nargs='+', type=str, required=True)
    parser.add_argument('--reference', type=int, required=True, default=None)
    parser.add_argument('--use_positions', type=int, required=True, default=None)
    parser.add_argument('--result_path', type=str, required=False, default=None)

    args = parser.parse_args()

    assert len(args.multicamcalibs) == len(args.poses), "Each multicamcalib set must have a set of poses"
    combine(args.multicamcalibs, args.poses,
            reference=args.reference, use_positions=args.use_positions, result_path=args.result_path)


def combine(multicamcalibs_in, poses_in, reference=0, use_positions=0, result_path=None):
    # Joins two calibration that where performed on the SAME set of frames with COMPATIBLE BOARDS (i.e. common poses)
    multicamcalibs = []
    poses = []
    frame_idxs = None
    for i_component, (multicamcalib, pose) in enumerate(zip(multicamcalibs_in, poses_in)):
        if isinstance(multicamcalib, str):
            multicamcalib = Path(multicamcalib)
        if isinstance(pose, str):
            pose = Path(pose)

        if isinstance(multicamcalib, Path):
            multicamcalib = multicamcalib.expanduser().resolve()
            multicamcalib = Camerasystem.load_dict(multicamcalib)
        if isinstance(pose, Path):
            pose = pose.expanduser().resolve()
            with open(pose, 'r') as f:
                pose = yaml.safe_load(f)

        multicamcalibs.append(deepcopy(multicamcalib))
        poses.append(deepcopy(pose))

        if frame_idxs is None:
            frame_idxs = set(pose["frame_idxs"][0])
        else:
            frame_idxs = frame_idxs.intersection(pose["frame_idxs"][0])

    assert len(frame_idxs)>0, "No common frames found."
    print(f"Found {len(frame_idxs)} common frames.")

    frame_idxs = np.array(list(frame_idxs))
    print(frame_idxs)
    Ts_b02wo = []  # Ideal base to worlds to be oriented
    for pose in poses:
        frame_mask = np.isin(pose["frame_idxs"][0], frame_idxs)
        Ts_b02wo.append(RigidTransform(rotation=np.array(pose["rvecs"])[frame_mask], translation=np.array(pose["tvecs"])[frame_mask]))

    T_b02wr = Ts_b02wo[reference]  # Ideal base to reference world

    for i_set, T_b02wo in enumerate(Ts_b02wo):
        T_wo2wr=T_b02wr*T_b02wo.inv()
        for calib in multicamcalibs[i_set]["calibs"]:
            T_wo2co = RigidTransform(rotation=calib["rvec_cam"],
                                    translation=calib["tvec_cam"],
                                    rotation_type="rotvec")
            T_wr2co = T_wo2co * T_wo2wr.inv()
            calib["rvec_cam"] = T_wr2co.get_rotation().as_rotvec()
            calib["tvec_cam"] = T_wr2co.get_translation()

    multicamcalib_final = None
    for i_set, multicamcalib in enumerate(multicamcalibs):
        if i_set == 0:
            multicamcalib_final = multicamcalib
            continue
        multicamcalib_final["calibs"]+=multicamcalib["calibs"]
        for k in ["board_params", "rec_file_names", "used_frames_ids", "vid_headers"]:
            multicamcalib_final["info"][k] += multicamcalib["info"][k]

    if result_path is None:
        result_path = Path(multicamcalibs_in[0]).parent / "multicamcalibration_combined"
    print(result_path)
    save_multicalibration(result_path, multicamcalib_final)

if __name__ == '__main__':
    main()
