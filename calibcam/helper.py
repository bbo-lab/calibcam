import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa


# Detection may not lie on a single line
def check_detections_nondegenerate(board_width, charuco_ids):
    charuco_ids = np.asarray(charuco_ids).ravel()

    # Not enough points
    if len(charuco_ids) < 5:
        # print(f"{len(charuco_ids)} charuco_ids are not enough!")
        return False

    # All points along one row (width)
    if charuco_ids[-1] < (np.floor(charuco_ids[0] / (board_width - 1)) + 1) * (
            board_width - 1):
        # print(f"{len(charuco_ids)} charuco_ids are in a row!: {charuco_ids}")
        return False

    # All points along one column (height)
    if np.all(np.mod(np.diff(charuco_ids), board_width - 1) == 0):
        # print(f"{len(charuco_ids)} charuco_ids are in a column!: {charuco_ids}")
        return False

    return True


def deepmerge_dicts(source, destination):
    """
    merges source into destination
    """

    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            deepmerge_dicts(value, node)
        else:
            destination[key] = value

    return destination

def make_corners_array(corners_all, ids_all, n_corners, frames_masks):
    used_frames_mask = np.any(frames_masks, axis=0)
    used_frame_idxs = np.where(used_frames_mask)[0]

    corners = np.empty(shape=(frames_masks.shape[0], used_frames_mask.sum(), n_corners, 2), dtype=np.float32)
    corners[:] = np.NaN
    for i_cam, frames_mask_cam in enumerate(frames_masks):
        frame_idxs_cam = np.where(frames_mask_cam)[0]

        for i_frame, f_idx in enumerate(used_frame_idxs):
            # print(ids_all[i_cam][i_frame].ravel())
            # print(corners[i_cam, f_idx].shape)
            # print(corners_all[i_cam][i_frame].shape)
            cam_fr_idx = np.where(frame_idxs_cam == f_idx)[0]
            if cam_fr_idx.size < 1:
                continue

            cam_fr_idx = int(cam_fr_idx)
            if ids_all is None:
                corners[i_cam, i_frame] = \
                    corners_all[i_cam][cam_fr_idx][:, 0, :]
            else:
                corners[i_cam, i_frame][ids_all[i_cam][cam_fr_idx].ravel(), :] = \
                    corners_all[i_cam][cam_fr_idx][:, 0, :]
    return corners


def corners_array_to_ragged(corners_array):
    ids_use = [np.where(~np.isnan(c[:, 1]))[0].astype(np.int32).reshape(-1, 1) for c in corners_array]
    corners_use = [c[i, :].reshape(-1, 1, 2) for c, i in zip(corners_array, ids_use)]

    return corners_use, ids_use


def build_v1_result(result):
    return {
        'A_fit': np.array([c['A'] for c in result['calibs']]),
        'k_fit': np.array([c['k'] for c in result['calibs']]),
        'rX1_fit': np.array([c['rvec_cam'] for c in result['calibs']]),
        'RX1_fit': R.from_rotvec(np.array([c['rvec_cam'] for c in result['calibs']])).as_matrix(),
        'tX1_fit': np.array([c['tvec_cam'] for c in result['calibs']]),
        'nCameras': len(result['calibs']),
        'version': 1.0,
    }


def combine_calib_with_board_params(calibs, rvecs_boards, tvecs_boards):
    for i_cam, calib in enumerate(calibs):
        calib['rvecs'] = rvecs_boards
        calib['tvecs'] = tvecs_boards

    return calibs
        
