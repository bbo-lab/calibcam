from svidreader import filtergraph
import imageio
from calibcam import board
from calibcam.calibrator_opts import get_default_opts, finalize_aruco_detector_opts
from calibcam.board import get_board_params
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def main(frame_num=1295):
    video = "/media/smb/soma-fs.ad01.caesar.de/bbo/bulk/bird_hellenthal_FrontField/20230630_20230630_FlightWindow/20230630/eye_setup/2023_06_30_16_15_22_calibration_20230630/2.mp4"
    pipeline = "[input_0]crop=size=1280x1024;permutate=map=/media/smb/soma-fs.ad01.caesar.de//bbo/users/cheekoti/data/junker-bird-data/frame_permutation/20230630/anat_eye/2023_06_30_16_15_22_calibration_20230630/permutation_2.csv:source=from:destination=to"
    reader = filtergraph.get_reader(video, backend='iio')
    reader = filtergraph.create_filtergraph_from_string([reader], pipeline)['out']
    test_frame = reader.get_data(frame_num)
    imageio.imwrite(f'test/sample_images_1/{frame_num}.png', test_frame)

    board_name = "/media/smb/soma-fs.ad01.caesar.de/bbo/projects/junker-bird/experiments/20230612_20230615_FlightWindow/calibrations/board_small"
    board_params = get_board_params(board_name)
    opts = get_default_opts()
    detection = opts['detection']
    detection['aruco_refine']['errorCorrectionRate'] = 0.1
    detection['aruco_interpolate']['minMarkers'] = 1

    detector_parameters = {  # SPOT for detector params
        'adaptiveThreshWinSizeMin': 23,
        'adaptiveThreshWinSizeMax': 43,
        'adaptiveThreshWinSizeStep': 5,

        'polygonalApproxAccuracyRate': 0.01,

        'perspectiveRemovePixelPerCell': 8,
        'maxErroneousBitsInBorderRate': 0.1,
        'errorCorrectionRate': 0.1,

        'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
        'cornerRefinementWinSize': 3,
        'cornerRefinementMaxIterations': 100,
        'cornerRefinementMinAccuracy': 0.05,
    }
    detection['aruco_detect']['parameters'] = detector_parameters
    detection['aruco_refine']['parameters'] = detector_parameters

    # corner detection
    corners, ids, rejected_img_points = \
        cv2.aruco.detectMarkers(test_frame,  # noqa
                                cv2.aruco.getPredefinedDictionary(board_params['dictionary_type']),  # noqa
                                **finalize_aruco_detector_opts(opts['detection']['aruco_detect']))
    # corner refinement
    corners_ref, ids_ref = \
        cv2.aruco.refineDetectedMarkers(test_frame,  # noqa
                                        board.make_board(board_params),
                                        corners,
                                        ids,
                                        rejected_img_points,
                                        **finalize_aruco_detector_opts(opts['detection']['aruco_refine']))[0:2]

    #print(corners_ref)
    # corner interpolation
    retval, charuco_corners, charuco_ids = \
        cv2.aruco.interpolateCornersCharuco(corners_ref,  # noqa
                                            ids_ref,
                                            test_frame,
                                            board.make_board(board_params),
                                            **opts['detection']['aruco_interpolate'])

    print(charuco_corners,charuco_ids)

    charuco_corners = charuco_corners.reshape(-1, 2)
    plt.figure()
    plt.imshow(test_frame)
    plt.plot(charuco_corners[:, 0], charuco_corners[:, 1], "x", c="red")
    for c_ref in corners_ref:
        c_ref = c_ref.reshape(-1, 2)
        plt.plot(c_ref[:, 0], c_ref[:, 1], "o", c="blue")