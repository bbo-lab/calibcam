import numpy as np
from calibcam import helper


def update_preoptim_2_0_to_2_1(preoptim, n_corners):
    if preoptim['version'] < 2.1:
        frames_masks = preoptim['info']['frames_masks'].astype(bool)
        calibs_single = preoptim['info']['other']['calibs_single']

        preoptim['info']['corners'] = helper.make_corners_array(preoptim['info']['corners'],
                                                                preoptim['info']['corner_ids'], n_corners, frames_masks)

        used_frames_ids = np.where(np.any(frames_masks, axis=0))[0]

        for i_cam, calib in enumerate(calibs_single):
            rvecs = calib["rvecs"]
            calib["rvecs"] = np.empty(shape=(len(used_frames_ids), 3))
            calib["rvecs"][:] = np.NaN
            calib["rvecs"][frames_masks[i_cam, used_frames_ids], :] = rvecs.reshape(-1, 3)

            tvecs = calib["tvecs"]
            calib["tvecs"] = np.empty(shape=(len(used_frames_ids), 3))
            calib["tvecs"][:] = np.NaN
            calib["tvecs"][frames_masks[i_cam, used_frames_ids], :] = tvecs.reshape(-1, 3)

        preoptim['info']['other']['calibs_single'] = calibs_single
        preoptim['info']['used_frames_ids'] = used_frames_ids

    if preoptim['version'] < 2.2:
        for i_cam, calib in enumerate(preoptim['info']['other']['calibs_single']):
            calib["k"] = calib["k"].ravel()

    return preoptim


def update_preoptim(preoptim, n_corners):
    return update_preoptim_2_0_to_2_1(preoptim, n_corners)
