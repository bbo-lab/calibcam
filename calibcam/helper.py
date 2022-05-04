import numpy as np
from .exceptions import *

def get_n_frames_from_reader(reader):
    n_frames = len(reader)  # len() may be Inf for formats where counting frames can be expensive
    if 1000000000000000 < n_frames:
        try:
            n_frames = reader.count_frames()
        except ValueError:
            print('Could not determine number of frames')
            raise UnsupportedFormatException

    return n_frames


def get_header_from_reader(reader):
    header = reader.get_meta_data()
    # Add required headers that are not normally part of standard video formats but are required information for a full calibration
    # TODO add option to supply this via options. Currently, compressed
    if "sensor" in header:
        header['offset'] = tuple(header['sensor']['offset'])
        header['sensorsize'] = tuple(header['sensor']['size'])
        del header['sensor']
    else:
        if 'offset' not in header:
            print("Setting offset to 0!")
            header['offset'] = tuple(np.asarray([0, 0]))

        if 'sensorsize' not in header:
            print("Inferring sensor size from image")
            header['sensorsize'] = tuple(reader.get_data(0).shape[0:2])

    return header


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


def save_multicalibration_to_matlabcode(result, path):
    result_path_text = path + '/multicalibration_matlab_mcl_gen.m'

    f = open(result_path_text, 'w')

    f.write('% INFO:\n')
    f.write('% line break: \n \n')
    f.write('% See: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html\n')
    f.write('\n\n\n')

    i = 'nCameras'
    n_cams = result[i]
    f.write('% number of cameras:\n')
    f.write('nCams = {:01d}\n'.format(n_cams))
    f.write('\n\n')

    i = 'recFileNames'
    f.write('% full path of used files (order equal to camera indexing):\n')
    f.write('f = { ...\n')
    for j in result[i]:
        f.write('\'' + str(j) + '\',...\n')
    f.write('}\n')
    f.write('\n\n')

    i = 'indexRefCam'
    f.write('% index of reference camera (starts at 0):\n')
    f.write('index_refCam = {:01d}\n'.format(result[i]))
    f.write('\n\n')

    i = 'A_fit'
    data_use = result[i]
    f.write('% camera matrices [f_x, c_x, f_y, c_y]:\n')
    f.write('A = [ ...\n')
    for i_cam in range(n_cams):
        for i_row in range(4):
            f.write(str(data_use[i_cam, i_row]) + ' ')
        if i_cam != n_cams - 1:
            f.write('; ...\n')
        else:
            f.write(']\n')
    f.write('\n\n')

    i = 'k_fit'
    data_use = result[i]
    f.write('% distortion coefficients [k_1, k_2, p_1, p_2, k_3]:\n')
    f.write('k = [ ...\n')
    for i_cam in range(n_cams):
        for i_row in range(5):
            f.write(str(data_use[i_cam, i_row]) + ' ')
        if i_cam != n_cams - 1:
            f.write('; ...\n')
        else:
            f.write(']\n')
    f.write('\n\n')

    i = 'RX1_fit'
    data_use = result[i]
    f.write('% rotation matrices to convert into coordinate system of the respective camera:\n')
    f.write('R = cat(3, ...\n')
    for i_cam in range(n_cams):
        f.write('[')
        for i_row in range(3):
            for i_col in range(3):
                f.write(str(data_use[i_cam, i_row, i_col]) + ' ')
            if i_row != 2:
                f.write('; ...\n')
            else:
                if i_cam != n_cams - 1:
                    f.write('], ...\n')
                else:
                    f.write('])\n')
    f.write('\n\n')

    i = 'tX1_fit'
    data_use = result[i]
    f.write('% translation vectors to convert into coordinate system of the respective camera (units in squares):\n')
    f.write('t = [ ...\n')
    for i_cam in range(n_cams):
        for i_row in range(3):
            f.write(str(data_use[i_cam, i_row]) + ' ')
        if i_cam != n_cams - 1:
            f.write('; ...\n')
        else:
            f.write(']\'\n')
    f.write('\n\n')

    i = 'headers'
    data_use = result[i]

    f.write('% sensor size in pixel:\n')
    f.write('sensorSize = [ ...\n')
    for i_cam in range(n_cams):
        for i_row in range(2):
            f.write(str(data_use[i_cam]['sensorsize'][i_row]) + ' ')
        if i_cam != n_cams - 1:
            f.write('; ...\n')
        else:
            f.write(']\n')
    f.write('\n\n')

    f.write('% offset in pixel:\n')
    f.write('offset = [ ...\n')
    for i_cam in range(n_cams):
        for i_row in range(2):
            f.write(str(data_use[i_cam]['offset'][i_row]) + ' ')
        if i_cam != n_cams - 1:
            f.write('; ...\n')
        else:
            f.write(']\n')
    f.write('\n\n')

    # optional:
    # f.write('% used width in pixel:\n')
    # f.write('width = [ ...\n')
    # for i_cam in range(nCams):
    #    f.write(str(data_use[i_cam]['w']) + ' ')
    #    if (i_cam != nCams - 1):
    #        f.write('; ...\n')
    #    else:
    #        f.write(']\n')
    # f.write('\n\n')
    #
    # f.write('% used height in pixel:\n')
    # f.write('height = [ ...\n')
    # for i_cam in range(nCams):
    #     f.write(str(data_use[i_cam]['h']) + ' ')
    #     if (i_cam != nCams - 1):
    #         f.write('; ...\n')
    #     else:
    #         f.write(']\n')
    # f.write('\n\n')

    f.write('% square size in cm:\n')
    f.write('square_size = {:.8f}\n'.format(result['square_size_real']))
    f.write('\n\n')

    f.write('% marker size in cm:\n')
    f.write('marker_size = {:.8f}\n'.format(result['marker_size_real']))
    f.write('\n\n')

    f.write('% scale factor in cm:\n')
    f.write('scale_factor = {:.8f}\n'.format(result['scale_factor']))
    f.write('\n\n')

    f.write(
        "[mc ,mcfn] = cameralib.helper.openCVToMCL(R,t,A,k,sensorSize,scale_factor,bbohelper.filesystem.filename(f))\n"
        "mcl = cameralib.MultiCamSetupModel.fromMCL(mcfn)\n"
        "mcl.save([mcfn(1:end-3) 'mat'])\n"
    )

    f.close()
    print('Saved multi camera calibration to file {:s}'.format(result_path_text))
    return