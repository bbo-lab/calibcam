from copy import deepcopy

import numpy as np
import scipy.stats as stats
from scipy.spatial.transform import Rotation as R  # noqa
from svidreader.video_supplier import VideoSupplier

from calibcamlib import Detections


# Detection may not lie on a single line
def check_detections_nondegenerate(board_width, charuco_ids, minimum_points=5):
    charuco_ids = np.asarray(charuco_ids).ravel()

    # Not enough points
    if len(charuco_ids) < minimum_points:
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


# def corners_array_to_ragged(markers, squeeze=True):
#     markers_list = []
#     for marker_coords_cam in markers["marker_coords"]:
#         marker_coords_used = []
#         frame_idxs_used = []
#         marker_ids_used = []
#         for i_frame, marker_coords_cam_frame in enumerate(marker_coords_cam):
#
#
#         marker_ids_used = [
#             markers["marker_ids"][np.where(~np.isnan(c[:, 1]))[0].astype(np.int32).reshape(-1)]
#             for c in marker_coords_cam]
#         marker_coords_used = [
#             c[]
#         ]
#
#         markers_list.append({
#             "marker_coords": corners,
#             "frame_idxs": frame_idxs_used,
#             "marker_ids": marker_ids_used,
#         })
#
#     ids_use = [np.where(~np.isnan(c[:, 1]))[0].astype(np.int32).reshape(-1) for c in corners_array]
#     corners_use = [c[i, :].astype(np.float32).reshape(-1, 1, corner_shape) for c, i in zip(corners_array, ids_use)]
#
#     return corners_use, ids_use


def build_v1_result(result):
    # TODO: should include xi
    return {
        'A_fit': np.array([c['A'] for c in result['calibs']]),
        'k_fit': np.array([c['k'] for c in result['calibs']]),
        'rX1_fit': np.array([c['rvec_cam'] for c in result['calibs']]),
        'RX1_fit': R.from_rotvec(np.array([c['rvec_cam'] for c in result['calibs']])).as_matrix(),
        'tX1_fit': np.array([c['tvec_cam'] for c in result['calibs']]),
        'nCameras': len(result['calibs']),
        'version': 1.0,
    }


def combine_calib_with_board_poses(calibs, rvecs_boards, tvecs_boards, copy=False):
    if copy:
        calibs = deepcopy(calibs)

    for i_cam, calib in enumerate(calibs):
        calib['rvecs'] = rvecs_boards
        calib['tvecs'] = tvecs_boards

    return calibs


def nearest_element(num_1: int, list_nums):
    dist = np.abs(np.asarray(list_nums) - num_1)
    return list_nums[np.argmin(dist)]


def combine_boards_to_points(boards, marker_ids):
    board_start_ids = [brd.get_board_ids()[0] for brd in boards]
    for brd in boards:
        print(brd.get_board_points()[:, np.newaxis].shape)
    board_points = [[brd.get_board_points()[:, np.newaxis] for brd in boards]]
    board_ids = [[np.arange(len(bp))+bsid for bp,bsid in zip(board_points[0],board_start_ids)]]

    print(len(board_points), len(board_points[0]), len(board_points[0][0]), len(board_points[0][0][0]))
    print(board_ids)
    board_coords = Detections.from_list(board_points, board_ids, return_dict=True)

    marker_mask = np.isin(board_coords["marker_ids"], marker_ids)
    print(marker_ids, marker_mask)
    return np.nanmean(board_coords["marker_coords"], axis=1)[0, marker_mask]


class RadialContrast(VideoSupplier):
    def __init__(self, reader, options={}, width=20, normalize=50, norm_mean=0.31):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.lib = options.get('lib', "cupy")
        if self.lib == 'cupy':
            import cupy as cp
            import cupyx.scipy.ndimage
            sqnorm = cp.fuse(RadialContrast.sqnorm(cp))
            conv = cupyx.scipy.ndimage.convolve
            self.xp = cp
        elif self.lib == 'jax':
            import jax
            sqnorm = jax.jit(RadialContrast.sqnorm(jax.numpy))
            conv = jax.scipy.signal.convolve
            self.xp = jax.numpy
        elif self.lib == 'nb':
            import numba as nb
            import scipy
            sqnorm = nb.jit(RadialContrast.sqnorm(np))
            conv = scipy.ndimage.convolve
            self.xp = np
        else:
            import scipy
            sqnorm = RadialContrast.sqnorm(np)
            conv = scipy.ndimage.convolve
            self.xp = np
        self.convolve = RadialContrast.get_convolve(self.xp, conv, sqnorm,
                                                    width=width, normalize=normalize, norm_mean=norm_mean)

    @staticmethod
    def get_convolve(xp, conv, sqnorm, width=20, normalize=np.nan, norm_mean=0.31):
        xx, yy = np.mgrid[-1:1:width * 1j, -1:1:width * 1j]
        w0 = np.sin(np.arctan2(yy, xx) * 2)
        w1 = np.cos(np.arctan2(yy, xx) * 2)
        d = np.sqrt(np.square(xx) + np.square(yy))
        mask = d < 1.1
        wpeak = xp.asarray((stats.norm.pdf(d, 0, 1) - norm_mean) * mask, dtype=xp.float32)
        mask = mask * stats.norm.pdf(d, 0, 1)
        w0 = w0 * mask
        w1 = w1 * mask
        w0 = xp.asarray(w0, dtype=xp.float32)
        w1 = xp.asarray(w1, dtype=xp.float32)
        if not np.isnan(normalize):
            w0 *= 1 / xp.sum(xp.abs(w0))
            w1 *= 1 / xp.sum(xp.abs(w1))
            wpeak *= normalize / xp.sum(xp.abs(wpeak))

        def convolve_impl(res):
            res = xp.sum(res, axis=2)
            res = sqnorm(conv(res, w0), conv(res, w1))
            res = xp.maximum(conv(res, wpeak), 0)
            return res

        return convolve_impl

    @staticmethod
    def sqnorm(xp):
        def f(gx, gy):
            gx = xp.square(gx)
            gy = xp.square(gy)
            res = gx + gy
            res = xp.sqrt(res)
            return res

        return f

    def read(self, index, force_type=np):
        img = self.inputs[0].read(index=index, force_type=self.xp)
        img = self.xp.asarray(img, dtype=self.xp.float32)
        img = self.convolve(img)
        img = self.xp.minimum(img, 255)
        return VideoSupplier.convert(self.xp.asarray(img, dtype=self.xp.uint8), force_type)
