# Converts array of rotation vectors to array of rotation matrices
# TODO check performance of this vs other solutions
def rodrigues2rotMats(r):
    nRes = len(r)
    theta = np.power(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2, 0.5)
    u = r / (theta + -np.abs(np.sign(theta)) + 1).reshape(nRes, 1)
    # row 1
    rotMat_00 = np.cos(theta) + u[:, 0] ** 2 * (1 - np.cos(theta))
    rotMat_01 = u[:, 0] * u[:, 1] * (1 - np.cos(theta)) - u[:, 2] * np.sin(theta)
    rotMat_02 = u[:, 0] * u[:, 2] * (1 - np.cos(theta)) + u[:, 1] * np.sin(theta)
    rotMat_0 = np.concatenate([rotMat_00.reshape(nRes, 1, 1),
                               rotMat_01.reshape(nRes, 1, 1),
                               rotMat_02.reshape(nRes, 1, 1)], 2)

    # row 2
    rotMat_10 = u[:, 0] * u[:, 1] * (1 - np.cos(theta)) + u[:, 2] * np.sin(theta)
    rotMat_11 = np.cos(theta) + u[:, 1] ** 2 * (1 - np.cos(theta))
    rotMat_12 = u[:, 1] * u[:, 2] * (1 - np.cos(theta)) - u[:, 0] * np.sin(theta)
    rotMat_1 = np.concatenate([rotMat_10.reshape(nRes, 1, 1),
                               rotMat_11.reshape(nRes, 1, 1),
                               rotMat_12.reshape(nRes, 1, 1)], 2)

    # row 3
    rotMat_20 = u[:, 0] * u[:, 2] * (1 - np.cos(theta)) - u[:, 1] * np.sin(theta)
    rotMat_21 = u[:, 1] * u[:, 2] * (1 - np.cos(theta)) + u[:, 0] * np.sin(theta)
    rotMat_22 = np.cos(theta) + u[:, 2] ** 2 * (1 - np.cos(theta))
    rotMat_2 = np.concatenate([rotMat_20.reshape(nRes, 1, 1),
                               rotMat_21.reshape(nRes, 1, 1),
                               rotMat_22.reshape(nRes, 1, 1)], 2)

    rotMat = np.concatenate([rotMat_0,
                             rotMat_1,
                             rotMat_2], 1)

    return rotMat

