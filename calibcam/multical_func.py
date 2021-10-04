import autograd.numpy as np
import re
import scipy
import yaml

def open_cv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    s = re.findall('\d+', mapping["dt"])
    nChannels = 1
    if (s):
        nChannels = int(s[0])
    mat.resize(mapping["rows"], mapping["cols"] * nChannels)
    return mat

def read_YAMLFile(fileName):
    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", open_cv_matrix)
    skip_lines = 1
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        result = yaml.load(fin.read())
    return result

def read_data(filePath, args):
    nCameras = args['nCameras']

    calib = dict()
    for i_camera in range(0, nCameras, 1):
        calib['cam{:01d}'.format(i_camera)] = \
            read_YAMLFile(filePath + '/{:02d}/calib.yml'.format(i_camera))

    return calib

def read_data_single(filePath, args):
    nCameras = args['nCameras']

    calib = dict()
    for i_camera in range(0, nCameras, 1):
        calib['cam{:01d}'.format(i_camera)] = \
            read_YAMLFile(filePath + '/{:02d}/calib_single.yml'.format(i_camera))

    return calib

def rodrigues2rotMat_single(r):
    theta = np.power(r[0]**2 + r[1]**2 + r[2]**2, 0.5)
    u = r / (theta + -np.abs(np.sign(theta)) + 1)
    # row 1
    rotMat_00 = np.cos(theta) + u[0]**2 * (1 - np.cos(theta))
    rotMat_01 = u[0] * u[1] * (1 - np.cos(theta)) - u[2] * np.sin(theta)
    rotMat_02 = u[0] * u[2] * (1 - np.cos(theta)) + u[1] * np.sin(theta)

    # row 2
    rotMat_10 = u[0] * u[1] * (1 - np.cos(theta)) + u[2] * np.sin(theta)
    rotMat_11 = np.cos(theta) + u[1]**2 * (1 - np.cos(theta))
    rotMat_12 = u[1] * u[2] * (1 - np.cos(theta)) - u[0] * np.sin(theta)

    # row 3
    rotMat_20 = u[0] * u[2] * (1 - np.cos(theta)) - u[1] * np.sin(theta)
    rotMat_21 = u[1] * u[2] * (1 - np.cos(theta)) + u[0] * np.sin(theta)
    rotMat_22 = np.cos(theta) + u[2]**2 * (1 - np.cos(theta))

    rotMat = np.array([[rotMat_00, rotMat_01, rotMat_02],
                       [rotMat_10, rotMat_11, rotMat_12],
                       [rotMat_20, rotMat_21, rotMat_22]])

    return rotMat

def rotMat2rodrigues_single(R):
    r = np.zeros((3, 1), dtype=float)
    K = (R - R.T) / 2
    r[0] = K[2, 1]
    r[1] = K[0, 2]
    r[2] = K[1, 0]

    if not(np.all(R == np.identity(3))):
        R_logm = scipy.linalg.logm(R)
        thetaM_1 = R_logm[2, 1] / (r[0] + np.equal(r[0], 0.0))
        thetaM_2 = R_logm[0, 2] / (r[1] + np.equal(r[1], 0.0))
        thetaM_3 = R_logm[1, 0] / (r[2] + np.equal(r[2], 0.0))
        thetaM = np.array([thetaM_1, thetaM_2, thetaM_3])

        theta = np.mean(thetaM[thetaM != 0.0])
        r = r * theta

    return r

def calc_xcam(calib, args):
    nCameras = args['nCameras']
    nPoses = args['nPoses']
    minDetectFeat = args['minDetectFeat']
    indexRefCam = args['indexRefCam']

    rcam = dict()
    tcam = dict()

    r11 = np.zeros((3, 1))
    rcam['r{:01d}{:01d}'.format(indexRefCam, indexRefCam)] = r11
    t11 = np.zeros((3, 1))
    tcam['t{:01d}{:01d}'.format(indexRefCam, indexRefCam)] = t11
    for i_camera in range(0, nCameras, 1):
        if (i_camera != indexRefCam):
            rX1 = np.zeros((3, 1), dtype=np.float64)
            RX1 = np.zeros((3, 3), dtype=float)
            tX1 = np.zeros((3, 1), dtype=np.float64)
            nPoses_use = 0
            for i_pose in range(0, nPoses, 1):
                idX = calib['cam{:01d}'.format(i_camera)]['charuco_ids'][i_pose]
                id1 = calib['cam{:01d}'.format(indexRefCam)]['charuco_ids'][i_pose]
                id_combi = np.intersect1d(idX, id1)
                if (id_combi.size >= minDetectFeat):
                    # RX1
                    rotVecX = calib['cam{:01d}'.format(i_camera)]['rotation_vectors'][nPoses_use].ravel()
                    RX = rodrigues2rotMat_single(rotVecX)
                    rotVec1 = calib['cam{:01d}'.format(indexRefCam)]['rotation_vectors'][nPoses_use].ravel()
                    R1 = rodrigues2rotMat_single(rotVec1)
                    RX1_add = np.dot(RX, R1.T)
                    RX1 += RX1_add
                    tX = calib['cam{:01d}'.format(i_camera)]['translation_vectors'][nPoses_use]
                    t1 = calib['cam{:01d}'.format(indexRefCam)]['translation_vectors'][nPoses_use]
                    tX1_add = (tX - np.dot(RX1_add, t1))
                    tX1 = tX1 + tX1_add
                    nPoses_use = nPoses_use + 1

            # Based on Curtis et al., A Note on Averaging Rotations (Lemma 2.2)
            u, s, vh = np.linalg.svd(RX1, full_matrices=True)
            RX1 = np.dot(u, vh)
            rX1 = rotMat2rodrigues_single(RX1)
            rcam['r{:01d}{:01d}'.format(i_camera, indexRefCam)] = rX1
            tX1 = tX1 / nPoses_use
            tcam['t{:01d}{:01d}'.format(i_camera, indexRefCam)] = tX1
    return rcam, tcam

def set_x0_objFunc(calib, args):
    nCameras = args['nCameras']
    nPoses = args['nPoses']
    kSize = args['kSize']
    ASize = args['ASize']
    rSize = args['rSize']
    tSize = args['tSize']
    nAllVars = args['nAllVars']
    indexRefCam = args['indexRefCam']

    x0 = np.zeros(nAllVars, dtype=float)
    i = 0
    A_useIndex = np.array([True, False, True,
                           False, True, True,
                           False, False, False], dtype=bool)
    rcam, tcam = calc_xcam(calib, args)
    # rX1
    for i_camera in range(0, nCameras, 1):
        if (i_camera != indexRefCam):
            rX1 = rcam['r{:01d}{:01d}'.format(i_camera, indexRefCam)]
            x0[i:i + rSize] = rX1.ravel()
            i = i + rSize
    # tX1
    for i_camera in range(0, nCameras, 1):
        if (i_camera != indexRefCam):
            tX1 = tcam['t{:01d}{:01d}'.format(i_camera, indexRefCam)]
            x0[i:i + tSize] = tX1.ravel()
            i = i + tSize
    # k
    for i_camera in range(0, nCameras, 1):
        k = calib['cam{:01d}'.format(i_camera)]['dist_coeffs'].ravel()
        x0[i:i + kSize] = k.ravel()
        i = i + kSize
    # A
    for i_camera in range(0, nCameras, 1):
        A = calib['cam{:01d}'.format(i_camera)]['camera_matrix'].ravel()[A_useIndex]
        x0[i:i + ASize] = A.ravel()
        i = i + ASize
    # r1
    i_pose_use = 0
    for i_pose in range(0, nPoses, 1):
        r1 = calib['cam{:01d}'.format(indexRefCam)]['rotation_vectors'][i_pose_use]
        x0[i:i + rSize] = r1.ravel()
        i_pose_use = i_pose_use + 1
        i = i + rSize
    # t1
    i_pose_use = 0
    for i_pose in range(0, nPoses, 1):
        t1 = calib['cam{:01d}'.format(indexRefCam)]['translation_vectors'][i_pose_use]
        x0[i:i + tSize] = t1.ravel()
        i_pose_use = i_pose_use + 1
        i = i + tSize

    return x0

def set_x0_objFunc_single(calib_single, args):
    nCameras = args['nCameras']
    nPoses_single = args['nPoses_single']
    rSize = args['rSize']
    tSize = args['tSize']
    nAllVars_single = args['nAllVars_single']

    x0_single = np.zeros(nAllVars_single, dtype=float)
    i = 0
    # rX1, tX1: not present in single calibration
    # k, A: already implemented in set_x0_objFunc
    for i_camera in range(0, nCameras, 1):
        key = 'cam{:01d}'.format(i_camera)
        for i_pose in range(0, nPoses_single[i_camera], 1):
            # r1
            r1 = calib_single[key]['rotation_vectors'][i_pose]
            x0_single[i:i + rSize] = r1.ravel()
            i = i + rSize
            # t1
            t1 = calib_single[key]['translation_vectors'][i_pose]
            x0_single[i:i + tSize] = t1.ravel()
            i = i + rSize

    return x0_single

def obj_fcn_free(x, args):
    nAllVars_all = args['nAllVars_all']
    x0_all = args['x0_all']
    free_para_all = args['free_para_all']

    x_all_use = np.zeros(nAllVars_all, dtype=float)
    x_all_use[free_para_all] = x
    x_all_use[~free_para_all] = x0_all[~free_para_all]
    obj_fcn_free_val = obj_fcn(x_all_use, args)

    return obj_fcn_free_val

def obj_fcn_jac_free(x, args):
    nAllVars_all = args['nAllVars_all']
    x0_all = args['x0_all']
    free_para_all = args['free_para_all']

    x_all_use = np.zeros(nAllVars_all, dtype=float)
    x_all_use[free_para_all] = x
    x_all_use[~free_para_all] = x0_all[~free_para_all]
    obj_fcn_jac_free_val = obj_fcn_jac(x_all_use, args)[:, free_para_all]

    return obj_fcn_jac_free_val

def map_calib2consts(calib, args):
    nCameras = args['nCameras']
    nPoses = args['nPoses']
    nFeatures = args['nFeatures']
    nRes = args['nRes']
    boardWidth = args['boardWidth']
    boardHeight = args['boardHeight']
    nFeatures = args['nFeatures']
    minDetectFeat = args['minDetectFeat']

    # M
    M_0 = np.repeat(np.arange(1, boardWidth).reshape(1, boardWidth-1), boardHeight-1, axis=0).ravel().reshape(nFeatures, 1)
    M_1 = np.repeat(np.arange(1, boardHeight), boardWidth-1, axis=0).reshape(nFeatures, 1)
    M_ini = np.concatenate([M_0, M_1], 1)

    M = np.zeros((nRes, 2), dtype=np.float64)
    m = np.zeros((nRes, 2), dtype=np.float64)
    delta = np.zeros(nRes, dtype=np.float64)
    index = np.zeros(nFeatures, dtype=bool)
    indexRes = 0
    for i_camera in range(0, nCameras, 1):
        for i_pose in range(0, nPoses, 1):
            res_index1 = indexRes * nFeatures
            indexRes = indexRes + 1
            res_index2 = indexRes * nFeatures
            # M
            M[res_index1:res_index2:1] = M_ini

            corners = calib['cam{:01d}'.format(i_camera)]['charuco_corners'][i_pose]
            idX = calib['cam{:01d}'.format(i_camera)]['charuco_ids'][i_pose]
            if (idX.size >= minDetectFeat):
                index[:] = False
                index[idX.T[0]] = True
                # m
                m[res_index1:res_index2:1][idX.T[0]] = corners
                # delta
                delta[res_index1:res_index2:1] = index.astype(np.float64)

    return M, m, delta

def map_calib2consts_single(calib_single, args):
    nCameras = args['nCameras']
    nPoses_single = args['nPoses_single']
    nFeatures = args['nFeatures']
    nRes_single = args['nRes_single']
    boardWidth = args['boardWidth']
    boardHeight = args['boardHeight']
    nFeatures = args['nFeatures']
    minDetectFeat = args['minDetectFeat']

    # M
    M_0 = np.repeat(np.arange(1, boardWidth).reshape(1, boardWidth-1), boardHeight-1, axis=0).ravel().reshape(nFeatures, 1)
    M_1 = np.repeat(np.arange(1, boardHeight), boardWidth-1, axis=0).reshape(nFeatures, 1)
    M_ini = np.concatenate([M_0, M_1], 1)

    M_single = np.zeros((nRes_single, 2), dtype=np.float64)
    m_single = np.zeros((nRes_single, 2), dtype=np.float64)
    delta_single = np.zeros(nRes_single, dtype=np.float64)
    index = np.zeros(nFeatures, dtype=bool)
    indexRes = 0
    for i_camera in range(0, nCameras, 1):
        for i_pose in range(0, nPoses_single[i_camera], 1):
            res_index1 = indexRes * nFeatures
            indexRes = indexRes + 1
            res_index2 = indexRes * nFeatures
            # M
            M_single[res_index1:res_index2:1] = M_ini
            corners = calib_single['cam{:01d}'.format(i_camera)]['charuco_corners'][i_pose]
            idX = calib_single['cam{:01d}'.format(i_camera)]['charuco_ids'][i_pose]
            index[:] = False
            index[idX.T[0]] = True
            if (idX.size >= minDetectFeat):
                # m
                m_single[res_index1:res_index2:1][idX.T[0]] = corners
                # delta
                delta_single[res_index1:res_index2:1] = index.astype(np.float64)

    return M_single, m_single, delta_single

def rodrigues2rotMat(r, nRes):
    theta = np.power(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2, 0.5)
    u = r / (theta + -np.abs(np.sign(theta)) + 1).reshape(nRes, 1)
    # row 1
    rotMat_00 = np.cos(theta) + u[:, 0]**2 * (1 - np.cos(theta))
    rotMat_01 = u[:, 0] * u[:, 1] * (1 - np.cos(theta)) - u[:, 2] * np.sin(theta)
    rotMat_02 = u[:, 0] * u[:, 2] * (1 - np.cos(theta)) + u[:, 1] * np.sin(theta)
    rotMat_0 = np.concatenate([rotMat_00.reshape(nRes, 1, 1),
                               rotMat_01.reshape(nRes, 1, 1),
                               rotMat_02.reshape(nRes, 1, 1)], 2)

    # row 2
    rotMat_10 = u[:, 0] * u[:, 1] * (1 - np.cos(theta)) + u[:, 2] * np.sin(theta)
    rotMat_11 = np.cos(theta) + u[:, 1]**2 * (1 - np.cos(theta))
    rotMat_12 = u[:, 1] * u[:, 2] * (1 - np.cos(theta)) - u[:, 0] * np.sin(theta)
    rotMat_1 = np.concatenate([rotMat_10.reshape(nRes, 1, 1),
                               rotMat_11.reshape(nRes, 1, 1),
                               rotMat_12.reshape(nRes, 1, 1)], 2)

    # row 3
    rotMat_20 = u[:, 0] * u[:, 2] * (1 - np.cos(theta)) - u[:, 1] * np.sin(theta)
    rotMat_21 = u[:, 1] * u[:, 2] * (1 - np.cos(theta)) + u[:, 0] * np.sin(theta)
    rotMat_22 = np.cos(theta) + u[:, 2]**2 * (1 - np.cos(theta))
    rotMat_2 = np.concatenate([rotMat_20.reshape(nRes, 1, 1),
                               rotMat_21.reshape(nRes, 1, 1),
                               rotMat_22.reshape(nRes, 1, 1)], 2)

    rotMat = np.concatenate([rotMat_0,
                             rotMat_1,
                             rotMat_2], 1)

    return rotMat

def map_m(rX1_0, rX1_1, rX1_2,
          tX1_0, tX1_1, tX1_2,
          r1_0, r1_1, r1_2,
          t1_0, t1_1, t1_2,
          M,
          nRes):
    # rX1
    rX1 = np.concatenate([rX1_0.reshape(nRes, 1),
                          rX1_1.reshape(nRes, 1),
                          rX1_2.reshape(nRes, 1)], 1)
    # tX1
    tX1 = np.concatenate([tX1_0.reshape(nRes, 1),
                          tX1_1.reshape(nRes, 1),
                          tX1_2.reshape(nRes, 1)], 1)
    # r1
    r1 = np.concatenate([r1_0.reshape(nRes, 1),
                         r1_1.reshape(nRes, 1),
                         r1_2.reshape(nRes, 1)], 1)
    # t1
    t1 = np.concatenate([t1_0.reshape(nRes, 1),
                         t1_1.reshape(nRes, 1),
                         t1_2.reshape(nRes, 1)], 1)


    RX1 = rodrigues2rotMat(rX1, nRes)
    R1 = rodrigues2rotMat(r1, nRes)

    # R1 * M + t1
    m_proj_0_1 = R1[:, 0, 0] * M[:, 0] + \
                 R1[:, 0, 1] * M[:, 1] + \
                 t1[:, 0]
    m_proj_1_1 = R1[:, 1, 0] * M[:, 0] + \
                 R1[:, 1, 1] * M[:, 1] + \
                 t1[:, 1]
    m_proj_2_1 = R1[:, 2, 0] * M[:, 0] + \
                 R1[:, 2, 1] * M[:, 1] + \
                 t1[:, 2]
    # RX1 * m_proj + tX1
    m_proj_0_2 = RX1[:, 0, 0] * m_proj_0_1 + \
                 RX1[:, 0, 1] * m_proj_1_1 + \
                 RX1[:, 0, 2] * m_proj_2_1 + \
                 tX1[:, 0]
    m_proj_1_2 = RX1[:, 1, 0] * m_proj_0_1 + \
                 RX1[:, 1, 1] * m_proj_1_1 + \
                 RX1[:, 1, 2] * m_proj_2_1 + \
                 tX1[:, 1]
    m_proj_2_2 = RX1[:, 2, 0] * m_proj_0_1 + \
                 RX1[:, 2, 1] * m_proj_1_1 + \
                 RX1[:, 2, 2] * m_proj_2_1 + \
                 tX1[:, 2]
    # m_proj / m_proj[2]
    x_pre = m_proj_0_2 / m_proj_2_2
    y_pre = m_proj_1_2 / m_proj_2_2
    # distort
    r2 = x_pre**2 + y_pre**2

    return x_pre, y_pre, r2

def calc_res_x(rX1_0, rX1_1, rX1_2,
               tX1_0, tX1_1, tX1_2,
               k_1, k_2, p_1, p_2, k_3,
               fx, cx, fy, cy,
               r1_0, r1_1, r1_2,
               t1_0, t1_1, t1_2,
               M, m, delta,
               nRes):
    x_pre, y_pre, r2 = map_m(rX1_0, rX1_1, rX1_2,
                             tX1_0, tX1_1, tX1_2,
                             r1_0, r1_1, r1_2,
                             t1_0, t1_1, t1_2,
                             M,
                             nRes)
    # distort
    x = x_pre * (1 + k_1 * r2 + k_2 * r2**2 + k_3 * r2**3) + \
        2 * p_1 * x_pre * y_pre + \
        p_2 * (r2 + 2 * x_pre**2)
    # A * m_proj
    x_post = x * fx + cx

    res_x = delta * (x_post - m[:, 0])

    return res_x

def calc_res_y(rX1_0, rX1_1, rX1_2,
               tX1_0, tX1_1, tX1_2,
               k_1, k_2, p_1, p_2, k_3,
               fx, cx, fy, cy,
               r1_0, r1_1, r1_2,
               t1_0, t1_1, t1_2,
               M, m, delta,
               nRes):
    x_pre, y_pre, r2 = map_m(rX1_0, rX1_1, rX1_2,
                             tX1_0, tX1_1, tX1_2,
                             r1_0, r1_1, r1_2,
                             t1_0, t1_1, t1_2,
                             M,
                             nRes)
    # distort
    y = y_pre * (1 + k_1 * r2 + k_2 * r2**2 + k_3 * r2**3) + \
        p_1 * (r2 + 2 * y_pre**2) + \
        2 * p_2 * x_pre * y_pre
    # A * m_proj
    y_post = y * fy + cy

    res_y = delta * (y_post - m[:, 1])

    return res_y

def calc_paras_from_x(x, args):
    nCameras = args['nCameras']
    nPoses = args['nPoses']

    kSize = args['kSize']
    ASize = args['ASize']
    rSize = args['rSize']
    tSize = args['tSize']

    indexRefCam = args['indexRefCam']

    index = 0
    # rX1
    rX1_ref = np.zeros((1, 3))
    rX1_use = x[index:index + rSize * (nCameras - 1)].reshape((nCameras - 1), rSize)
    rX1 = np.concatenate([rX1_use[:indexRefCam, :], rX1_ref, rX1_use[indexRefCam:, :]], 0)
    index = index + rSize * (nCameras - 1)
    # tX1
    tX1_ref = np.zeros((1, 3))
    tX1_use = x[index:index + tSize * (nCameras - 1)].reshape((nCameras - 1), tSize)
    tX1 = np.concatenate([tX1_use[:indexRefCam, :], tX1_ref, tX1_use[indexRefCam:, :]], 0)
    index = index + tSize * (nCameras - 1)
    # k
    k = x[index:index + kSize * nCameras].reshape(nCameras, kSize)
    index = index + kSize * nCameras
    # A
    A = x[index:index + ASize * nCameras].reshape(nCameras, ASize)
    index = index + ASize * nCameras
    # r1
    r1 = x[index:index + rSize * nPoses].reshape(nPoses, rSize)
    index = index + rSize * nPoses
    # t1
    t1 = x[index:index + tSize * nPoses].reshape(nPoses, rSize)

    return rX1, tX1, k, A, r1, t1

def calc_paras_from_x_single(x_single, args):
    nCameras = args['nCameras']
    nPoses_single = args['nPoses_single']

    rSize = args['rSize']
    tSize = args['tSize']

    nAllVars_single = args['nAllVars_single']

    nPoses_single_max = np.max(nPoses_single)
    r1_single = np.zeros((nCameras, nPoses_single_max, rSize), dtype=np.float64)
    t1_single = np.zeros((nCameras, nPoses_single_max, tSize), dtype=np.float64)

    mask = np.zeros(nAllVars_single, dtype=bool)
    mask_index = 0
    for i_camera in range(0, nCameras, 1):
        mask[:] = False
        mask[mask_index:mask_index + (rSize + tSize) * nPoses_single[i_camera]] = True
        mask_index = mask_index + (rSize + tSize) * nPoses_single[i_camera]

        x_use = x_single[mask].reshape(nPoses_single[i_camera], 2, rSize)
        # r1
        r1_single[i_camera][:nPoses_single[i_camera]] = x_use[:, 0]
        # t1
        t1_single[i_camera][:nPoses_single[i_camera]] = x_use[:, 1]

    return r1_single, t1_single

def calc_paras_from_x_single2(x_single, args):
    nCameras = args['nCameras']
    nPoses_single = args['nPoses_single']

    rSize = args['rSize']
    tSize = args['tSize']

    nAllVars_single = args['nAllVars_single']

    r1_single = list()
    t1_single = list()

    mask = np.zeros(nAllVars_single, dtype=bool)
    mask_index = 0
    for i_camera in range(0, nCameras, 1):
        mask[:] = False
        mask[mask_index:mask_index + (rSize + tSize) * nPoses_single[i_camera]] = True
        mask_index = mask_index + (rSize + tSize) * nPoses_single[i_camera]

        x_use = x_single[mask].reshape(nPoses_single[i_camera], 2, rSize)
        # r1
        r1_single.append(list())
        r1_single[i_camera].append(x_use[:, 0].squeeze())
        # t1
        t1_single.append(list())
        t1_single[i_camera].append(x_use[:, 1].squeeze())

    return r1_single, t1_single

def map_r2R(r):
    n_iter = np.size(r, 0)

    R = np.zeros((n_iter, 3, 3), dtype=float)
    for i in range(0, n_iter, 1):
        R[i] = rodrigues2rotMat_single(r[i, :])

    return R

def map_xOpt2xRes(x_opt, args):
    nCameras = args['nCameras']
    nPoses = args['nPoses']
    nPoses_single = args['nPoses_single']
    nFeatures = args['nFeatures']
    nRes = args['nRes']
    nRes_single = args['nRes_single']

    kSize = args['kSize']
    ASize = args['ASize']
    rSize = args['rSize']
    tSize = args['tSize']

    nVars = args['nVars']
    nAllVars = args['nAllVars']

    # multi
    x_res = np.zeros((nRes, nVars), dtype=float)

    rX1, tX1, k, A, r1, t1 = calc_paras_from_x(x_opt[:nAllVars], args)

    indexRes = 0
    for i_camera in range(0, nCameras, 1):
        rX1_use = np.repeat(rX1[i_camera].reshape(1, rSize), nFeatures, axis=0)
        tX1_use = np.repeat(tX1[i_camera].reshape(1, tSize), nFeatures, axis=0)
        k_use = np.repeat(k[i_camera].reshape(1, kSize), nFeatures, axis=0)
        A_use = np.repeat(A[i_camera].reshape(1, ASize), nFeatures, axis=0)
        for i_pose in range(0, nPoses, 1):
            res_index1 = indexRes * nFeatures
            indexRes = indexRes + 1
            res_index2 = indexRes * nFeatures
            i = 0
            # rX1
            x_res[res_index1:res_index2, i:i + rSize] = rX1_use
            i = i + rSize
            # tX1
            x_res[res_index1:res_index2, i:i + tSize] = tX1_use
            i = i + tSize
            # k
            x_res[res_index1:res_index2, i:i + kSize] = k_use
            i = i + kSize
            # A
            x_res[res_index1:res_index2, i:i + ASize] = A_use
            i = i + ASize
            # r1
            r1_use = np.repeat(r1[i_pose].reshape(1, rSize), nFeatures, axis=0)
            x_res[res_index1:res_index2, i:i + rSize] = r1_use
            i = i + rSize
            # t1
            t1_use = np.repeat(t1[i_pose].reshape(1, tSize), nFeatures, axis=0)
            x_res[res_index1:res_index2, i:i + tSize] = t1_use

    # single
    x_res_single = np.zeros((nRes_single, nVars), dtype=float)

    r1_single, t1_single = calc_paras_from_x_single(x_opt[nAllVars:], args)

    indexRes = 0
    for i_camera in range(0, nCameras, 1):
        k_use = np.repeat(k[i_camera].reshape(1, kSize), nFeatures, axis=0)
        A_use = np.repeat(A[i_camera].reshape(1, ASize), nFeatures, axis=0)
        for i_pose in range(0, nPoses_single[i_camera], 1):
            res_index1 = indexRes * nFeatures
            indexRes = indexRes + 1
            res_index2 = indexRes * nFeatures
            # rX1, tx1
            i = rSize + tSize
            # k
            x_res_single[res_index1:res_index2, i:i + kSize] = k_use
            i = i + kSize
            # A
            x_res_single[res_index1:res_index2, i:i + ASize] = A_use
            i = i + ASize
            # r1
            r1_single_use = np.repeat(r1_single[i_camera][i_pose].reshape(1, rSize), nFeatures, axis=0)
            x_res_single[res_index1:res_index2, i:i + rSize] = r1_single_use
            i = i + rSize
            # t1
            t1_single_use = np.repeat(t1_single[i_camera][i_pose].reshape(1, tSize), nFeatures, axis=0)
            x_res_single[res_index1:res_index2, i:i + tSize] = t1_single_use

    x_res_all = np.concatenate([x_res, x_res_single], 0)

    return x_res_all

def obj_fcn(x_opt, args):
    M = args['M']
    m = args['m']
    delta = args['delta']
    M_single = args['M_single']
    m_single = args['m_single']
    delta_single = args['delta_single']

    nRes = args['nRes']
    nRes_single = args['nRes_single']

    x_res = map_xOpt2xRes(x_opt, args)

    # multi
    obj_fcn_val_x = calc_res_x(x_res[:nRes, 0], x_res[:nRes, 1], x_res[:nRes, 2],
                               x_res[:nRes, 3], x_res[:nRes, 4], x_res[:nRes, 5],
                               x_res[:nRes, 6], x_res[:nRes, 7], x_res[:nRes, 8], x_res[:nRes, 9], x_res[:nRes, 10],
                               x_res[:nRes, 11], x_res[:nRes, 12], x_res[:nRes, 13], x_res[:nRes, 14],
                               x_res[:nRes, 15], x_res[:nRes, 16], x_res[:nRes, 17],
                               x_res[:nRes, 18], x_res[:nRes, 19], x_res[:nRes, 20],
                               M, m, delta,
                               nRes)
    obj_fcn_val_y = calc_res_y(x_res[:nRes, 0], x_res[:nRes, 1], x_res[:nRes, 2],
                               x_res[:nRes, 3], x_res[:nRes, 4], x_res[:nRes, 5],
                               x_res[:nRes, 6], x_res[:nRes, 7], x_res[:nRes, 8], x_res[:nRes, 9], x_res[:nRes, 10],
                               x_res[:nRes, 11], x_res[:nRes, 12], x_res[:nRes, 13], x_res[:nRes, 14],
                               x_res[:nRes, 15], x_res[:nRes, 16], x_res[:nRes, 17],
                               x_res[:nRes, 18], x_res[:nRes, 19], x_res[:nRes, 20],
                               M, m, delta,
                               nRes)
    # single
    obj_fcn_val_x_single = calc_res_x(x_res[nRes:, 0], x_res[nRes:, 1], x_res[nRes:, 2],
                                      x_res[nRes:, 3], x_res[nRes:, 4], x_res[nRes:, 5],
                                      x_res[nRes:, 6], x_res[nRes:, 7], x_res[nRes:, 8], x_res[nRes:, 9], x_res[nRes:, 10],
                                      x_res[nRes:, 11], x_res[nRes:, 12], x_res[nRes:, 13], x_res[nRes:, 14],
                                      x_res[nRes:, 15], x_res[nRes:, 16], x_res[nRes:, 17],
                                      x_res[nRes:, 18], x_res[nRes:, 19], x_res[nRes:, 20],
                                      M_single, m_single, delta_single,
                                      nRes_single)
    obj_fcn_val_y_single = calc_res_y(x_res[nRes:, 0], x_res[nRes:, 1], x_res[nRes:, 2],
                                      x_res[nRes:, 3], x_res[nRes:, 4], x_res[nRes:, 5],
                                      x_res[nRes:, 6], x_res[nRes:, 7], x_res[nRes:, 8], x_res[nRes:, 9], x_res[nRes:, 10],
                                      x_res[nRes:, 11], x_res[nRes:, 12], x_res[nRes:, 13], x_res[nRes:, 14],
                                      x_res[nRes:, 15], x_res[nRes:, 16], x_res[nRes:, 17],
                                      x_res[nRes:, 18], x_res[nRes:, 19], x_res[nRes:, 20],
                                      M_single, m_single, delta_single,
                                      nRes_single)

    obj_fcn_val = np.concatenate([obj_fcn_val_x, obj_fcn_val_y,
                                  obj_fcn_val_x_single, obj_fcn_val_y_single], 0)

    return obj_fcn_val

def obj_fcn_jac(x_opt, args):
    nRes = args['nRes']

    x_res = map_xOpt2xRes(x_opt, args)

    x_res_multi = x_res[:nRes]
    x_res_single = x_res[nRes:]
    jac_multi = fill_jac_multi(x_res_multi, args)
    jac_single = fill_jac_single(x_res_single, args)
    jac = np.concatenate([jac_multi, jac_single], 0)

    return jac

def fill_jac_multi(x_res, args):
    nCameras = args['nCameras']
    nPoses = args['nPoses']
    nFeatures = args['nFeatures']
    nRes = args['nRes']

    kSize = args['kSize']
    ASize = args['ASize']
    rSize = args['rSize']
    tSize = args['tSize']

    nAllVars_all = args['nAllVars_all']

    indexRefCam = args['indexRefCam']

    M = args['M']
    m = args['m']
    delta = args['delta']

    resPerCam = nPoses * nFeatures
    obj_fcn_jac_val_x = np.zeros((nRes, nAllVars_all), dtype=np.float64)
    obj_fcn_jac_val_y = np.zeros((nRes, nAllVars_all), dtype=np.float64)

    # MULTI
    args_i = 0
    out_i = 0
    indexRef = np.ones(nRes, dtype=bool)
    indexRef1 = indexRefCam * resPerCam
    indexRef2 = (indexRefCam + 1) * resPerCam
    indexRef[indexRef1:indexRef2] = False
    # rX1
    for i in range(0, rSize, 1):
        df_dx = args['jac_x'][i + args_i](x_res[indexRef, 0], x_res[indexRef, 1], x_res[indexRef, 2],
                                          x_res[indexRef, 3], x_res[indexRef, 4], x_res[indexRef, 5],
                                          x_res[indexRef, 6], x_res[indexRef, 7], x_res[indexRef, 8], x_res[indexRef, 9], x_res[indexRef, 10],
                                          x_res[indexRef, 11], x_res[indexRef, 12], x_res[indexRef, 13], x_res[indexRef, 14],
                                          x_res[indexRef, 15], x_res[indexRef, 16], x_res[indexRef, 17],
                                          x_res[indexRef, 18], x_res[indexRef, 19], x_res[indexRef, 20],
                                          M[indexRef], m[indexRef], delta[indexRef],
                                          nRes-resPerCam).reshape(nCameras - 1, resPerCam)
        obj_fcn_jac_val_x[indexRef, i + out_i:rSize * (nCameras - 1) + out_i:rSize] = scipy.linalg.block_diag(*df_dx).T
        df_dx = args['jac_y'][i + args_i](x_res[indexRef, 0], x_res[indexRef, 1], x_res[indexRef, 2],
                                          x_res[indexRef, 3], x_res[indexRef, 4], x_res[indexRef, 5],
                                          x_res[indexRef, 6], x_res[indexRef, 7], x_res[indexRef, 8], x_res[indexRef, 9], x_res[indexRef, 10],
                                          x_res[indexRef, 11], x_res[indexRef, 12], x_res[indexRef, 13], x_res[indexRef, 14],
                                          x_res[indexRef, 15], x_res[indexRef, 16], x_res[indexRef, 17],
                                          x_res[indexRef, 18], x_res[indexRef, 19], x_res[indexRef, 20],
                                          M[indexRef], m[indexRef], delta[indexRef],
                                          nRes-resPerCam).reshape(nCameras - 1, resPerCam)
        obj_fcn_jac_val_y[indexRef, i + out_i:rSize * (nCameras - 1) + out_i:rSize] = scipy.linalg.block_diag(*df_dx).T
    args_i = args_i + rSize
    out_i = out_i + (nCameras - 1) * rSize
    # tX1
    for i in range(0, tSize, 1):
        df_dx = args['jac_x'][i + args_i](x_res[indexRef, 0], x_res[indexRef, 1], x_res[indexRef, 2],
                                          x_res[indexRef, 3], x_res[indexRef, 4], x_res[indexRef, 5],
                                          x_res[indexRef, 6], x_res[indexRef, 7], x_res[indexRef, 8], x_res[indexRef, 9], x_res[indexRef, 10],
                                          x_res[indexRef, 11], x_res[indexRef, 12], x_res[indexRef, 13], x_res[indexRef, 14],
                                          x_res[indexRef, 15], x_res[indexRef, 16], x_res[indexRef, 17],
                                          x_res[indexRef, 18], x_res[indexRef, 19], x_res[indexRef, 20],
                                          M[indexRef], m[indexRef], delta[indexRef],
                                          nRes-resPerCam).reshape(nCameras - 1, resPerCam)
        obj_fcn_jac_val_x[indexRef, i + out_i:tSize * (nCameras - 1) + out_i:tSize] = scipy.linalg.block_diag(*df_dx).T
        df_dx = args['jac_y'][i + args_i](x_res[indexRef, 0], x_res[indexRef, 1], x_res[indexRef, 2],
                                          x_res[indexRef, 3], x_res[indexRef, 4], x_res[indexRef, 5],
                                          x_res[indexRef, 6], x_res[indexRef, 7], x_res[indexRef, 8], x_res[indexRef, 9], x_res[indexRef, 10],
                                          x_res[indexRef, 11], x_res[indexRef, 12], x_res[indexRef, 13], x_res[indexRef, 14],
                                          x_res[indexRef, 15], x_res[indexRef, 16], x_res[indexRef, 17],
                                          x_res[indexRef, 18], x_res[indexRef, 19], x_res[indexRef, 20],
                                          M[indexRef], m[indexRef], delta[indexRef],
                                          nRes-resPerCam).reshape(nCameras - 1, resPerCam)
        obj_fcn_jac_val_y[indexRef, i + out_i:tSize * (nCameras - 1) + out_i:tSize] = scipy.linalg.block_diag(*df_dx).T
    args_i = args_i + tSize
    out_i = out_i + (nCameras - 1) * tSize
    # k
    for i in range(0, kSize, 1):
        df_dx = args['jac_x'][i + args_i](x_res[:, 0], x_res[:, 1], x_res[:, 2],
                                          x_res[:, 3], x_res[:, 4], x_res[:, 5],
                                          x_res[:, 6], x_res[:, 7], x_res[:, 8], x_res[:, 9], x_res[:, 10],
                                          x_res[:, 11], x_res[:, 12], x_res[:, 13], x_res[:, 14],
                                          x_res[:, 15], x_res[:, 16], x_res[:, 17],
                                          x_res[:, 18], x_res[:, 19], x_res[:, 20],
                                          M, m, delta,
                                          nRes).reshape(nCameras, resPerCam)
        obj_fcn_jac_val_x[:, i + out_i:kSize * nCameras + out_i:kSize] = scipy.linalg.block_diag(*df_dx).T
        df_dx = args['jac_y'][i + args_i](x_res[:, 0], x_res[:, 1], x_res[:, 2],
                                          x_res[:, 3], x_res[:, 4], x_res[:, 5],
                                          x_res[:, 6], x_res[:, 7], x_res[:, 8], x_res[:, 9], x_res[:, 10],
                                          x_res[:, 11], x_res[:, 12], x_res[:, 13], x_res[:, 14],
                                          x_res[:, 15], x_res[:, 16], x_res[:, 17],
                                          x_res[:, 18], x_res[:, 19], x_res[:, 20],
                                          M, m, delta,
                                          nRes).reshape(nCameras, resPerCam)
        obj_fcn_jac_val_y[:, i + out_i:kSize * nCameras + out_i:kSize] = scipy.linalg.block_diag(*df_dx).T
    args_i = args_i + kSize
    out_i = out_i + nCameras * kSize
    # A
    ASize_use = int(ASize / 2)
    for i in range(0, ASize_use, 1):
        df_dx = args['jac_x'][i + args_i](x_res[:, 0], x_res[:, 1], x_res[:, 2],
                                          x_res[:, 3], x_res[:, 4], x_res[:, 5],
                                          x_res[:, 6], x_res[:, 7], x_res[:, 8], x_res[:, 9], x_res[:, 10],
                                          x_res[:, 11], x_res[:, 12], x_res[:, 13], x_res[:, 14],
                                          x_res[:, 15], x_res[:, 16], x_res[:, 17],
                                          x_res[:, 18], x_res[:, 19], x_res[:, 20],
                                          M, m, delta,
                                          nRes).reshape(nCameras, resPerCam)
        obj_fcn_jac_val_x[:, i + out_i:ASize * nCameras + out_i:ASize] = scipy.linalg.block_diag(*df_dx).T
        df_dx = args['jac_y'][i + ASize_use + args_i](x_res[:, 0], x_res[:, 1], x_res[:, 2],
                                          x_res[:, 3], x_res[:, 4], x_res[:, 5],
                                          x_res[:, 6], x_res[:, 7], x_res[:, 8], x_res[:, 9], x_res[:, 10],
                                          x_res[:, 11], x_res[:, 12], x_res[:, 13], x_res[:, 14],
                                          x_res[:, 15], x_res[:, 16], x_res[:, 17],
                                          x_res[:, 18], x_res[:, 19], x_res[:, 20],
                                          M, m, delta,
                                          nRes).reshape(nCameras, resPerCam)
        obj_fcn_jac_val_y[:, i + ASize_use + out_i:ASize * nCameras + out_i:ASize] = scipy.linalg.block_diag(*df_dx).T
    args_i = args_i + ASize
    out_i = out_i + nCameras * ASize

    index_local = np.ones((nPoses, nFeatures), dtype=bool)
    index_global_sub = scipy.linalg.block_diag(*index_local).T
    index_global = np.tile(index_global_sub, (nCameras, 1))

    # r1
    for i in range(0, rSize, 1):
        df_dx = args['jac_x'][i + args_i](x_res[:, 0], x_res[:, 1], x_res[:, 2],
                                          x_res[:, 3], x_res[:, 4], x_res[:, 5],
                                          x_res[:, 6], x_res[:, 7], x_res[:, 8], x_res[:, 9], x_res[:, 10],
                                          x_res[:, 11], x_res[:, 12], x_res[:, 13], x_res[:, 14],
                                          x_res[:, 15], x_res[:, 16], x_res[:, 17],
                                          x_res[:, 18], x_res[:, 19], x_res[:, 20],
                                          M, m, delta,
                                          nRes)
        obj_fcn_jac_val_x[:, i + out_i:rSize * nPoses + out_i:rSize][index_global] = df_dx
        df_dx = args['jac_y'][i + args_i](x_res[:, 0], x_res[:, 1], x_res[:, 2],
                                          x_res[:, 3], x_res[:, 4], x_res[:, 5],
                                          x_res[:, 6], x_res[:, 7], x_res[:, 8], x_res[:, 9], x_res[:, 10],
                                          x_res[:, 11], x_res[:, 12], x_res[:, 13], x_res[:, 14],
                                          x_res[:, 15], x_res[:, 16], x_res[:, 17],
                                          x_res[:, 18], x_res[:, 19], x_res[:, 20],
                                          M, m, delta,
                                          nRes)
        obj_fcn_jac_val_y[:, i + out_i:rSize * nPoses + out_i:rSize][index_global] = df_dx
    args_i = args_i + rSize
    out_i = out_i + nPoses * rSize
    # t1
    for i in range(0, tSize, 1):
        df_dx = args['jac_x'][i + args_i](x_res[:, 0], x_res[:, 1], x_res[:, 2],
                                          x_res[:, 3], x_res[:, 4], x_res[:, 5],
                                          x_res[:, 6], x_res[:, 7], x_res[:, 8], x_res[:, 9], x_res[:, 10],
                                          x_res[:, 11], x_res[:, 12], x_res[:, 13], x_res[:, 14],
                                          x_res[:, 15], x_res[:, 16], x_res[:, 17],
                                          x_res[:, 18], x_res[:, 19], x_res[:, 20],
                                          M, m, delta,
                                          nRes)
        obj_fcn_jac_val_x[:, i + out_i:tSize * nPoses + out_i:tSize][index_global] = df_dx
        df_dx = args['jac_y'][i + args_i](x_res[:, 0], x_res[:, 1], x_res[:, 2],
                                          x_res[:, 3], x_res[:, 4], x_res[:, 5],
                                          x_res[:, 6], x_res[:, 7], x_res[:, 8], x_res[:, 9], x_res[:, 10],
                                          x_res[:, 11], x_res[:, 12], x_res[:, 13], x_res[:, 14],
                                          x_res[:, 15], x_res[:, 16], x_res[:, 17],
                                          x_res[:, 18], x_res[:, 19], x_res[:, 20],
                                          M, m, delta,
                                          nRes)
        obj_fcn_jac_val_y[:, i + out_i:tSize * nPoses + out_i:tSize][index_global] = df_dx

    obj_fcn_jac_val = np.concatenate([obj_fcn_jac_val_x, obj_fcn_jac_val_y], 0)

    return obj_fcn_jac_val

def fill_jac_single(x_res, args):
    nCameras = args['nCameras']
    nPoses_single = args['nPoses_single']
    nFeatures = args['nFeatures']
    nRes_single = args['nRes_single']

    kSize = args['kSize']
    ASize = args['ASize']
    rSize = args['rSize']
    tSize = args['tSize']

    nAllVars = args['nAllVars']
    nAllVars_all = args['nAllVars_all']

    M_single = args['M_single']
    m_single = args['m_single']
    delta_single = args['delta_single']

    resPerCam_single = nPoses_single * nFeatures
    obj_fcn_jac_val_x = np.zeros((nRes_single, nAllVars_all), dtype=np.float64)
    obj_fcn_jac_val_y = np.zeros((nRes_single, nAllVars_all), dtype=np.float64)

    mask = np.zeros(nRes_single, dtype=bool)
    for i_camera in range(0, nCameras, 1):
        args_i = rSize + tSize
        out_i = (nCameras - 1) * (rSize + tSize)

        mask[:] = False
        mask_index1 = np.sum(resPerCam_single[:i_camera])
        mask_index2 = np.sum(resPerCam_single[:i_camera + 1])
        mask[mask_index1:mask_index2] = True
        x_use = x_res[mask, :]
        M_use = M_single[mask]
        m_use = m_single[mask]
        delta_use = delta_single[mask]

        # k
        for i in range(0, kSize, 1):
            df_dx = args['jac_x'][i + args_i](x_use[:, 0], x_use[:, 1], x_use[:, 2],
                                              x_use[:, 3], x_use[:, 4], x_use[:, 5],
                                              x_use[:, 6], x_use[:, 7], x_use[:, 8], x_use[:, 9], x_use[:, 10],
                                              x_use[:, 11], x_use[:, 12], x_use[:, 13], x_use[:, 14],
                                              x_use[:, 15], x_use[:, 16], x_use[:, 17],
                                              x_use[:, 18], x_use[:, 19], x_use[:, 20],
                                              M_use, m_use, delta_use,
                                              resPerCam_single[i_camera])
            obj_fcn_jac_val_x[mask, i + out_i + i_camera * kSize] = df_dx

            df_dx = args['jac_y'][i + args_i](x_use[:, 0], x_use[:, 1], x_use[:, 2],
                                              x_use[:, 3], x_use[:, 4], x_use[:, 5],
                                              x_use[:, 6], x_use[:, 7], x_use[:, 8], x_use[:, 9], x_use[:, 10],
                                              x_use[:, 11], x_use[:, 12], x_use[:, 13], x_use[:, 14],
                                              x_use[:, 15], x_use[:, 16], x_use[:, 17],
                                              x_use[:, 18], x_use[:, 19], x_use[:, 20],
                                              M_use, m_use, delta_use,
                                              resPerCam_single[i_camera])
            obj_fcn_jac_val_y[mask, i + out_i + i_camera * kSize] = df_dx

        args_i = args_i + kSize
        out_i = out_i + nCameras * kSize
        
        # A
        # assumes no skew!
        ASize_use = int(ASize / 2)
        for i in range(0, ASize_use, 1):
            df_dx = args['jac_x'][i + args_i](x_use[:, 0], x_use[:, 1], x_use[:, 2],
                                              x_use[:, 3], x_use[:, 4], x_use[:, 5],
                                              x_use[:, 6], x_use[:, 7], x_use[:, 8], x_use[:, 9], x_use[:, 10],
                                              x_use[:, 11], x_use[:, 12], x_use[:, 13], x_use[:, 14],
                                              x_use[:, 15], x_use[:, 16], x_use[:, 17],
                                              x_use[:, 18], x_use[:, 19], x_use[:, 20],
                                              M_use, m_use, delta_use,
                                              resPerCam_single[i_camera])
            obj_fcn_jac_val_x[mask, i + out_i + i_camera * ASize] = df_dx

            df_dx = args['jac_y'][i + args_i + ASize_use](x_use[:, 0], x_use[:, 1], x_use[:, 2],
                                              x_use[:, 3], x_use[:, 4], x_use[:, 5],
                                              x_use[:, 6], x_use[:, 7], x_use[:, 8], x_use[:, 9], x_use[:, 10],
                                              x_use[:, 11], x_use[:, 12], x_use[:, 13], x_use[:, 14],
                                              x_use[:, 15], x_use[:, 16], x_use[:, 17],
                                              x_use[:, 18], x_use[:, 19], x_use[:, 20],
                                              M_use, m_use, delta_use,
                                              resPerCam_single[i_camera])
            obj_fcn_jac_val_y[mask, i + out_i + i_camera * ASize + ASize_use] = df_dx
            
        args_i = args_i + ASize

        for i in range(0, rSize, 1):
            df_dx = args['jac_x'][i + args_i](x_use[:, 0], x_use[:, 1], x_use[:, 2],
                                              x_use[:, 3], x_use[:, 4], x_use[:, 5],
                                              x_use[:, 6], x_use[:, 7], x_use[:, 8], x_use[:, 9], x_use[:, 10],
                                              x_use[:, 11], x_use[:, 12], x_use[:, 13], x_use[:, 14],
                                              x_use[:, 15], x_use[:, 16], x_use[:, 17],
                                              x_use[:, 18], x_use[:, 19], x_use[:, 20],
                                              M_use, m_use, delta_use,
                                              int(resPerCam_single[i_camera])).reshape(nPoses_single[i_camera], nFeatures)
            obj_fcn_jac_val_x[mask, \
                              nAllVars + np.sum(nPoses_single[:i_camera]) * (rSize + tSize) + i: \
                              nAllVars + np.sum(nPoses_single[:i_camera + 1]) * (rSize + tSize) + i:
                              rSize + tSize] = scipy.linalg.block_diag(*df_dx).T

            df_dx = args['jac_y'][i + args_i](x_use[:, 0], x_use[:, 1], x_use[:, 2],
                                              x_use[:, 3], x_use[:, 4], x_use[:, 5],
                                              x_use[:, 6], x_use[:, 7], x_use[:, 8], x_use[:, 9], x_use[:, 10],
                                              x_use[:, 11], x_use[:, 12], x_use[:, 13], x_use[:, 14],
                                              x_use[:, 15], x_use[:, 16], x_use[:, 17],
                                              x_use[:, 18], x_use[:, 19], x_use[:, 20],
                                              M_use, m_use, delta_use,
                                              int(resPerCam_single[i_camera])).reshape(nPoses_single[i_camera], nFeatures)
            obj_fcn_jac_val_y[mask, \
                              nAllVars + np.sum(nPoses_single[:i_camera]) * (rSize + tSize) + i: \
                              nAllVars + np.sum(nPoses_single[:i_camera + 1]) * (rSize + tSize) + i:
                              rSize + tSize] = scipy.linalg.block_diag(*df_dx).T

        args_i = args_i + rSize

        # t1
        for i in range(0, tSize, 1):
            df_dx = args['jac_x'][i + args_i](x_use[:, 0], x_use[:, 1], x_use[:, 2],
                                              x_use[:, 3], x_use[:, 4], x_use[:, 5],
                                              x_use[:, 6], x_use[:, 7], x_use[:, 8], x_use[:, 9], x_use[:, 10],
                                              x_use[:, 11], x_use[:, 12], x_use[:, 13], x_use[:, 14],
                                              x_use[:, 15], x_use[:, 16], x_use[:, 17],
                                              x_use[:, 18], x_use[:, 19], x_use[:, 20],
                                              M_use, m_use, delta_use,
                                              int(resPerCam_single[i_camera])).reshape(nPoses_single[i_camera], nFeatures)
            obj_fcn_jac_val_x[mask, \
                              nAllVars + np.sum(nPoses_single[:i_camera]) * (rSize + tSize) + i + rSize: \
                              nAllVars + np.sum(nPoses_single[:i_camera + 1]) * (rSize + tSize) + i + rSize:
                              rSize + tSize] = scipy.linalg.block_diag(*df_dx).T

            df_dx = args['jac_y'][i + args_i](x_use[:, 0], x_use[:, 1], x_use[:, 2],
                                              x_use[:, 3], x_use[:, 4], x_use[:, 5],
                                              x_use[:, 6], x_use[:, 7], x_use[:, 8], x_use[:, 9], x_use[:, 10],
                                              x_use[:, 11], x_use[:, 12], x_use[:, 13], x_use[:, 14],
                                              x_use[:, 15], x_use[:, 16], x_use[:, 17],
                                              x_use[:, 18], x_use[:, 19], x_use[:, 20],
                                              M_use, m_use, delta_use,
                                              int(resPerCam_single[i_camera])).reshape(nPoses_single[i_camera], nFeatures)
            obj_fcn_jac_val_y[mask, \
                              nAllVars + np.sum(nPoses_single[:i_camera]) * (rSize + tSize) + i + rSize: \
                              nAllVars + np.sum(nPoses_single[:i_camera + 1]) * (rSize + tSize) + i + rSize:
                              rSize + tSize] = scipy.linalg.block_diag(*df_dx).T


    obj_fcn_jac_val = np.concatenate([obj_fcn_jac_val_x, obj_fcn_jac_val_y], 0)

    return obj_fcn_jac_val

def save_multicalibration_to_matlabcode(result,path):
    resultPath_text = path + '/multicalibration_matlab_mcl_gen.m'

    f = open(resultPath_text, 'w')

    f.write('% INFO:\n')
    f.write('% line break: \ n \n')
    f.write('% useful webpage: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html\n')
    f.write('\n\n\n')

    i = 'nCameras'
    nCams = result[i]
    f.write('% number of cameras:\n')
    f.write('nCams = {:01d}\n'.format(nCams))
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
    for i_cam in range(nCams):
        for i_row in range(4):
            f.write(str(data_use[i_cam, i_row]) + ' ')
        if (i_cam != nCams - 1):
            f.write('; ...\n')
        else:
            f.write(']\n')
    f.write('\n\n')

    i = 'k_fit'
    data_use = result[i]
    f.write('% distortion coefficients [k_1, k_2, p_1, p_2, k_3]:\n')
    f.write('k = [ ...\n')
    for i_cam in range(nCams):
        for i_row in range(5):
            f.write(str(data_use[i_cam, i_row]) + ' ')
        if (i_cam != nCams - 1):
            f.write('; ...\n')
        else:
            f.write(']\n')
    f.write('\n\n')

    i = 'RX1_fit'
    data_use = result[i]
    f.write('% rotation matrices to convert into coordinate system of the respective camera:\n')
    f.write('R = cat(3, ...\n')
    for i_cam in range(nCams):
        f.write('[')
        for i_row in range(3):
            for i_col in range(3):
                f.write(str(data_use[i_cam, i_row, i_col]) + ' ')
            if (i_row != 2):
                f.write('; ...\n')
            else:
                if (i_cam != nCams - 1):
                    f.write('], ...\n')
                else:
                    f.write('])\n')
    f.write('\n\n')

    i = 'tX1_fit'
    data_use = result[i]
    f.write('% translation vectors to convert into coordinate system of the respective camera (units in squares):\n')
    f.write('t = [ ...\n')
    for i_cam in range(nCams):
        for i_row in range(3):
            f.write(str(data_use[i_cam, i_row]) + ' ')
        if (i_cam != nCams - 1):
            f.write('; ...\n')
        else:
            f.write(']\'\n')
    f.write('\n\n')

    i = 'headers'
    data_use = result[i]
    
    f.write('% sensor size in pixel:\n')
    f.write('sensorSize = [ ...\n')
    for i_cam in range(nCams):
        for i_row in range(2):
            f.write(str(data_use[i_cam]['sensorsize'][i_row]) + ' ')
        if (i_cam != nCams - 1):
            f.write('; ...\n')
        else:
            f.write(']\n')
    f.write('\n\n')
    
    f.write('% offset in pixel:\n')
    f.write('offset = [ ...\n')
    for i_cam in range(nCams):
        for i_row in range(2):
            f.write(str(data_use[i_cam]['offset'][i_row]) + ' ')
        if (i_cam != nCams - 1):
            f.write('; ...\n')
        else:
            f.write(']\n')
    f.write('\n\n')
    
    # optional:
    #f.write('% used width in pixel:\n')
    #f.write('width = [ ...\n')
    #for i_cam in range(nCams):
        #f.write(str(data_use[i_cam]['w']) + ' ')
        #if (i_cam != nCams - 1):
            #f.write('; ...\n')
        #else:
            #f.write(']\n')
    #f.write('\n\n')
    
    #f.write('% used height in pixel:\n')
    #f.write('height = [ ...\n')
    #for i_cam in range(nCams):
        #f.write(str(data_use[i_cam]['h']) + ' ')
        #if (i_cam != nCams - 1):
            #f.write('; ...\n')
        #else:
            #f.write(']\n')
    #f.write('\n\n')

    f.write('% square size in cm:\n')
    f.write('square_size = {:.8f}\n'.format(result['square_size_real']))
    f.write('\n\n')

    f.write('% marker size in cm:\n')
    f.write('marker_size = {:.8f}\n'.format(result['marker_size_real']))
    f.write('\n\n')
    
    f.write('[mc ,mcfn] = cameralib.helper.openCVToMCL(R,t,A,k,sensorSize,square_size,bbohelper.filesystem.filename(f))\n'
            'mcl = cameralib.MultiCamSetupModel.fromMCL(mcfn)\n'
            "mcl.save([mcfn(1:end-3) 'mat'])\n"
            )

    f.close()
    print('Saved multi camera calibration to file {:s}'.format(resultPath_text))
    return
                

